# file 2

import os, os.path as osp, numpy as np, torch, cv2
from PIL import Image
from scipy.signal import find_peaks
from roma.utils import get_tuple_transform_ops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def log_strategy_action(message: str):
    print(f"[STRATEGY] {message}", flush=True)


class ActiveLearningStrategy:
    def __init__(self, args, cycle: int, data_root, split: str, idx_root=None, rng_seed: int = 784) -> None:
        self.data_root = data_root
        self.idx_root = idx_root or osp.join(self.data_root, "Idx_files")
        self.job_name = getattr(args, "job_name", "run")
        self.split = split
        self.strategy = getattr(args, "strategy", "coreset")
        self.cycle = int(cycle)
        self.train_pool_idx = self._load_idx(split)
        self.preseed_idx = self._load_idx("preseed_idx", required=False)
        self.train_current_idx = np.empty(0, dtype=int)
        if self.cycle > 0:
            prev_stem = f"{self.job_name}_cycle{self.cycle-1}"
            prev_path = self._idx_path(prev_stem)
            if osp.exists(prev_path):
                self.train_current_idx = np.load(prev_path).astype(int)
        self.rng_seed = int(rng_seed)
        self.rng = np.random.default_rng(self.rng_seed + self.cycle)
        log_strategy_action(
            f"{self.job_name} cycle {self.cycle}: strategy={self.strategy}, "
            f"pool={self.train_pool_idx.size}, preseed={self.preseed_idx.size}, "
            f"current={self.train_current_idx.size}"
        )

    def _save_hs_cert_plot(self, mu: float, sigma: float, tau: float, stem: str) -> None:
        out_dir = osp.dirname(__file__)
        fname = f"{stem}_hs_cert.png"
        out_path = osp.join(out_dir, fname)
        plt.figure(figsize=(5, 3))
        plt.axis("off")
        text = [
            f"{self.job_name} cycle {self.cycle}",
            f"mu   = {mu:.4f}",
            f"sigma = {sigma:.4f}",
            f"tau  = {tau:.4f}",
        ]
        plt.text(
            0.02,
            0.85,
            "\n".join(text),
            fontsize=12,
            verticalalignment="top",
            family="monospace",
        )
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        log_strategy_action(f"Saved hs_cert plot to {out_path}.")

    def _idx_path(self, name: str) -> str:
        fname = name if name.endswith(".npy") else f"{name}.npy"
        return osp.join(self.idx_root, fname)

    def _load_idx(self, name: str, required: bool = True) -> np.ndarray:
        path = self._idx_path(name)
        if not osp.exists(path):
            if required:
                raise FileNotFoundError(path)
            return np.empty(0, dtype=int)
        data = np.load(path).astype(int)
        log_strategy_action(f"Loaded index file {path} with {data.size} entries.")
        return data

    def _seed_indices(self) -> np.ndarray:
        parts = []
        if self.train_current_idx.size > 0:
            parts.append(self.train_current_idx.astype(int))
        if self.preseed_idx.size > 0:
            parts.append(self.preseed_idx.astype(int))
        if not parts:
            return np.empty(0, dtype=int)
        return np.unique(np.concatenate(parts, axis=0))

    def _ensure_preseed(self, idx: np.ndarray) -> np.ndarray:
        if (
            self.cycle == 0
            and self.preseed_idx.size > 0
            and self.strategy not in ("preseed", "full")
        ):
            if idx.size == 0:
                return self.preseed_idx.astype(int)
            return np.unique(np.concatenate([idx.astype(int), self.preseed_idx.astype(int)]))
        return idx

    def _seed_embeddings(self, encoder, device, tform):
        seed_idx = self._seed_indices()
        if seed_idx.size == 0:
            return None
        emb_list = [
            self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in seed_idx
        ]
        seed_embs = torch.stack(emb_list, dim=0).float()
        seed_embs = seed_embs / (seed_embs.norm(dim=1, keepdim=True) + 1e-8)
        return seed_embs

    def _seed_embeddings_raw(self, encoder, device, tform):
        seed_idx = self._seed_indices()
        if seed_idx.size == 0:
            return None
        emb_list = [
            self._pair_embedding_raw(encoder, device, tform, *self._idx_to_paths(int(i))) for i in seed_idx
        ]
        seed_embs = torch.stack(emb_list, dim=0).float()
        return seed_embs

    def _budget_for_cycle(self) -> int:
        schedule = {0: 10, 1: 20, 2: 40}
        budget = int(schedule.get(self.cycle, list(schedule.values())[-1]))
        log_strategy_action(f"Budget for cycle {self.cycle}: {budget} samples.")
        return budget

    def remaining(self) -> np.ndarray:
        return np.setdiff1d(self.train_pool_idx, self.train_current_idx, assume_unique=False)

    def _idx_to_paths(self, idx: int):
        a = osp.join(self.data_root, f"pair{int(idx)}_1.jpg")
        b = osp.join(self.data_root, f"pair{int(idx)}_2.jpg")
        if not osp.isfile(a) or not osp.isfile(b):
            raise FileNotFoundError(f"missing pair for {idx}")
        return a, b

    @torch.no_grad()
    def _pair_embedding_scale1(self, encoder, device, tform, a_path, b_path):
        imA = Image.open(a_path).convert("RGB")
        imB = Image.open(b_path).convert("RGB")
        xA, xB = tform((imA, imB))
        X = torch.cat([xA[None], xB[None]], dim=0).to(device, non_blocking=True)
        pyr = encoder(X, upsample=False)
        f = pyr[1]
        vA = f[0].mean(dim=(1, 2))
        vA = vA / (vA.norm() + 1e-8)
        vB = f[1].mean(dim=(1, 2))
        vB = vB / (vB.norm() + 1e-8)
        emb = torch.cat([vA, vB], dim=0)
        emb = emb / (emb.norm() + 1e-8)
        return emb.float().cpu()

    @torch.no_grad()
    def _pair_embedding_raw(self, encoder, device, tform, a_path, b_path):
        imA = Image.open(a_path).convert("RGB")
        imB = Image.open(b_path).convert("RGB")
        xA, xB = tform((imA, imB))
        X = torch.cat([xA[None], xB[None]], dim=0).to(device, non_blocking=True)
        pyr = encoder(X, upsample=False)
        f = pyr[1]
        vA = f[0].mean(dim=(1, 2))
        vB = f[1].mean(dim=(1, 2))
        emb = torch.cat([vA, vB], dim=0)
        return emb.float().cpu()

    @torch.no_grad()
    def _kcenter_from_vecs(self, X: torch.Tensor, k: int, seed_X: torch.Tensor | None = None) -> np.ndarray:
        X = X.float().cpu()
        if seed_X is not None and seed_X.numel() > 0:
            seed_X = seed_X.to(dtype=X.dtype, device=X.device)
        N = X.shape[0]
        if seed_X is None or seed_X.numel() == 0:
            rng = np.random.default_rng(self.rng_seed + self.cycle)
            init = int(rng.integers(0, N))
            picked = [init]
            dmin = torch.cdist(X, X[init:init+1]).squeeze(1)
            iters = k - 1
        else:
            dmin = torch.cdist(X, seed_X).min(dim=1).values
            picked = []
            iters = k
        for _ in range(max(0, iters)):
            j = int(torch.argmax(dmin).item())
            picked.append(j)
            dnew = torch.cdist(X, X[j:j+1]).squeeze(1)
            dmin = torch.minimum(dmin, dnew)
        return np.array(picked, dtype=int)

    @torch.no_grad()
    def _dpp_from_vecs(self, X: torch.Tensor, k: int) -> np.ndarray:
        X = X.float().cpu()
        N, D = X.shape
        if k <= 0 or N == 0:
            return np.empty(0, dtype=int)
        R = X.clone()
        selected = []
        selected_mask = torch.zeros(N, dtype=torch.bool)
        steps = min(k, N)
        for _ in range(steps):
            norms = (R * R).sum(dim=1)
            norms[selected_mask] = -1.0
            j = int(torch.argmax(norms).item())
            if norms[j] <= 0:
                break
            selected.append(j)
            selected_mask[j] = True
            b = R[j].unsqueeze(0)
            denom = float((b @ b.t()).item())
            if denom <= 0:
                continue
            proj = (R @ b.t()) / denom
            R = R - proj * b
        if len(selected) == 0:
            return np.empty(0, dtype=int)
        return np.array(selected, dtype=int)

    @torch.no_grad()
    def random(self, k: int) -> np.ndarray:
        avail = self.remaining()
        if avail.size == 0 or k <= 0:
            return np.empty(0, dtype=int)
        k = min(int(k), avail.size)
        idx = self.rng.choice(avail, size=k, replace=False)
        log_strategy_action(f"Random strategy picked {idx.size} indices out of {avail.size} available.")
        return idx.astype(int)

    @torch.no_grad()
    def weighted_tau_dpp(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        emb_list = []
        for i in avail:
            emb_list.append(self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))))
        cand_embs = torch.stack(emb_list, dim=0).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        hs_vals = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(dense_matches, dense_certainty, 5000, thresh_score=0.05)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 8:
                hs_vals.append(0.0)
                continue
            with Image.open(a_path) as imA:
                w1, h1 = imA.size
            with Image.open(b_path) as imB:
                w2, h2 = imB.size
            A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
            B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
            g = np.random.default_rng(1234)
            Hs = []
            subset = min(2000, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(50):
                sel = g.choice(A_px.shape[0], size=subset, replace=False) if A_px.shape[0] > subset else np.arange(A_px.shape[0])
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999)
                if H is not None and abs(H[2, 2]) > 1e-12:
                    Hs.append(H / (H[2, 2] + 1e-12))
            if len(Hs) < 2:
                hs_vals.append(0.0)
                continue
            Hs = np.stack(Hs, axis=0)
            c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
            warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
            s = float(warped.std(axis=0).mean())
            hs_vals.append(s)
        hs_vals = np.asarray(hs_vals, dtype=float)
        hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
        s_min, s_max = float(hs_cert.min()), float(hs_cert.max())
        hs_cert_norm = (hs_cert - s_min) / (s_max - s_min + 1e-8)
        mu = float(hs_cert_norm.mean())
        sigma = float(hs_cert_norm.std())
        tau = mu + sigma
        out_dir = osp.join(self.data_root, "weighted_tau_dpp_plots")
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        summary = [
            f"{self.job_name} cycle {self.cycle}",
            f"mu    = {mu:.4f}",
            f"sigma = {sigma:.4f}",
            f"tau   = {tau:.4f}",
        ]
        ax.text(0.02, 0.85, "\n".join(summary), va="top", ha="left", fontsize=12, family="monospace")
        fig.tight_layout()
        out_path = osp.join(out_dir, f"cycle{self.cycle}_tau_{tau:.4f}.png")
        fig.savefig(out_path)
        plt.close(fig)
        scale = (hs_cert_norm ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        R = cand_embs * scale_t
        picked_local = self._dpp_from_vecs(R, k=k)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def weighted_dpp_tau1(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        emb_list = []
        for i in avail:
            emb_list.append(self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))))
        cand_embs = torch.stack(emb_list, dim=0).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        hs_vals = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(dense_matches, dense_certainty, 5000, thresh_score=0.05)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 8:
                hs_vals.append(0.0)
                continue
            with Image.open(a_path) as imA:
                w1, h1 = imA.size
            with Image.open(b_path) as imB:
                w2, h2 = imB.size
            A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
            B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
            g = np.random.default_rng(1234)
            Hs = []
            subset = min(2000, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(50):
                sel = g.choice(A_px.shape[0], size=subset, replace=False) if A_px.shape[0] > subset else np.arange(A_px.shape[0])
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999)
                if H is not None and abs(H[2, 2]) > 1e-12:
                    Hs.append(H / (H[2, 2] + 1e-12))
            if len(Hs) < 2:
                hs_vals.append(0.0)
                continue
            Hs = np.stack(Hs, axis=0)
            c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
            warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
            s = float(warped.std(axis=0).mean())
            hs_vals.append(s)
        hs_vals = np.asarray(hs_vals, dtype=float)
        hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
        s_min, s_max = float(hs_cert.min()), float(hs_cert.max())
        hs_cert_norm = (hs_cert - s_min) / (s_max - s_min + 1e-8)
        scale = hs_cert_norm.astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        R = cand_embs * scale_t
        picked_local = self._dpp_from_vecs(R, k=k)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def coreset(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        seed_embs = self._seed_embeddings(encoder, device, tform)
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        picked_local = self._kcenter_from_vecs(cand_embs, k=k, seed_X=seed_embs)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def coreset2(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        seed_embs = self._seed_embeddings_raw(encoder, device, tform)
        cand_embs = torch.stack(
            [self._pair_embedding_raw(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        picked_local = self._kcenter_from_vecs(cand_embs, k=k, seed_X=seed_embs)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def roma_homography_stability(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        g = np.random.default_rng(1234)
        scores = []
        for idx in avail:
            a_path, b_path = self._idx_to_paths(int(idx))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(dense_matches, dense_certainty, 5000, thresh_score=0.05)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 8:
                scores.append((int(idx), 0.0))
                continue
            with Image.open(a_path) as imA:
                w1, h1 = imA.size
            with Image.open(b_path) as imB:
                w2, h2 = imB.size
            A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
            B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
            Hs = []
            subset = min(2000, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(50):
                sel = g.choice(A_px.shape[0], size=subset, replace=False) if A_px.shape[0] > subset else np.arange(A_px.shape[0])
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999)
                if H is not None and abs(H[2, 2]) > 1e-12:
                    Hs.append(H / (H[2, 2] + 1e-12))
            if len(Hs) < 2:
                scores.append((int(idx), 0.0))
                continue
            Hs = np.stack(Hs, axis=0)
            c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
            warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
            s = float(warped.std(axis=0).mean())
            scores.append((int(idx), s))
        scores.sort(key=lambda t: t[1], reverse=True)
        picked = np.array([i for i, _ in scores[:k]], dtype=int)
        return picked

    @torch.no_grad()
    def kcenter_uncertainty_weighted(self, model_for_uncertainty, k: int, lambda_u: float = 1.0) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        seed_embs = self._seed_embeddings(encoder, device, tform)
        hs_vals = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(dense_matches, dense_certainty, 5000, thresh_score=0.05)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 8:
                hs_vals.append(0.0)
                continue
            with Image.open(a_path) as imA:
                w1, h1 = imA.size
            with Image.open(b_path) as imB:
                w2, h2 = imB.size
            A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
            B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
            g = np.random.default_rng(1234)
            Hs = []
            subset = min(2000, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(50):
                sel = g.choice(A_px.shape[0], size=subset, replace=False) if A_px.shape[0] > subset else np.arange(A_px.shape[0])
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999)
                if H is not None and abs(H[2, 2]) > 1e-12:
                    Hs.append(H / (H[2, 2] + 1e-12))
            if len(Hs) < 2:
                hs_vals.append(0.0)
                continue
            Hs = np.stack(Hs, axis=0)
            c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
            warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
            s = float(warped.std(axis=0).mean())
            hs_vals.append(s)
        hs_vals = np.asarray(hs_vals, dtype=float)
        hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
        mu = float(hs_cert.mean())
        sigma = float(hs_cert.std())
        tau = mu + sigma
        stem = f"{self.job_name}_cycle{self.cycle}_kcenter"
        self._save_hs_cert_plot(mu, sigma, tau, stem)
        u_norm = 1.0 - hs_cert
        scale = ((1 + u_norm) ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def tau_weighted_embedding(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        encoder = getattr(model_for_uncertainty, "encoder", model_for_uncertainty)
        device = next(encoder.parameters()).device
        encoder.eval()
        Ht = getattr(model_for_uncertainty, "h_resized", 14 * 8 * 5)
        Wt = getattr(model_for_uncertainty, "w_resized", 14 * 8 * 5)
        tform = get_tuple_transform_ops(resize=(Ht, Wt), normalize=True, clahe=False)
        seed_embs = self._seed_embeddings_raw(encoder, device, tform)
        cand_embs = torch.stack(
            [self._pair_embedding_raw(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        hs_vals = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(dense_matches, dense_certainty, 5000, thresh_score=0.05)
            sm = sparse_matches.detach().cpu().numpy()
            if sm.shape[0] < 8:
                hs_vals.append(0.0)
                continue
            with Image.open(a_path) as imA:
                w1, h1 = imA.size
            with Image.open(b_path) as imB:
                w2, h2 = imB.size
            A_px = np.stack((w1 * (sm[:, 0] + 1) / 2 - 0.5, h1 * (sm[:, 1] + 1) / 2 - 0.5), axis=1)
            B_px = np.stack((w2 * (sm[:, 2] + 1) / 2 - 0.5, h2 * (sm[:, 3] + 1) / 2 - 0.5), axis=1)
            g = np.random.default_rng(1234)
            Hs = []
            subset = min(2000, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(50):
                sel = g.choice(A_px.shape[0], size=subset, replace=False) if A_px.shape[0] > subset else np.arange(A_px.shape[0])
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999)
                if H is not None and abs(H[2, 2]) > 1e-12:
                    Hs.append(H / (H[2, 2] + 1e-12))
            if len(Hs) < 2:
                hs_vals.append(0.0)
                continue
            Hs = np.stack(Hs, axis=0)
            c = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
            warped = np.stack([cv2.perspectiveTransform(c, H).reshape(4, 2) for H in Hs], axis=0)
            s = float(warped.std(axis=0).mean())
            hs_vals.append(s)
        hs_vals = np.asarray(hs_vals, dtype=float)
        hs_cert = 1.0 / (1.0 + np.maximum(hs_vals, 0.0))
        s_min, s_max = float(hs_cert.min()), float(hs_cert.max())
        hs_cert_norm = (hs_cert - s_min) / (s_max - s_min + 1e-8)
        mu = float(hs_cert_norm.mean())
        sigma = float(hs_cert_norm.std())
        tau = mu + sigma
        stem = f"{self.job_name}_cycle{self.cycle}"
        self._save_hs_cert_plot(mu, sigma, tau, stem)
        scale = (hs_cert_norm ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        picked = avail[picked_local]
        return picked.astype(int)

    def promote(self, new_idx: np.ndarray) -> None:
        new_idx = np.asarray(new_idx, dtype=int)
        if new_idx.size == 0:
            log_strategy_action(f"No new indices to promote for {self.job_name} cycle {self.cycle}.")
            return
        self.train_current_idx = np.unique(np.concatenate([self.train_current_idx, new_idx]))
        log_strategy_action(
            f"Promoted {new_idx.size} indices; total labeled set now {self.train_current_idx.size}."
        )

    def get_train_idx(self, model_for_uncertainty=None) -> str:
        if self.strategy == "full":
            new_idx = self.train_pool_idx.astype(int)
        elif self.strategy == "preseed":
            if self.preseed_idx.size == 0:
                raise ValueError("preseed_idx.npy missing")
            new_idx = self.preseed_idx.astype(int)
        else:
            k = self._budget_for_cycle()
            log_strategy_action(f"Running {self.strategy} strategy with budget {k}.")
            if self.strategy == "random":
                new_idx = self.random(k=k)
            elif self.strategy == "coreset" and model_for_uncertainty is not None:
                new_idx = self.coreset(model_for_uncertainty, k=k)
            elif self.strategy == "coreset2" and model_for_uncertainty is not None:
                new_idx = self.coreset2(model_for_uncertainty, k=k)
            elif self.strategy == "roma_homography_stability" and model_for_uncertainty is not None:
                new_idx = self.roma_homography_stability(model_for_uncertainty, k=k)
            elif self.strategy in ("kcenter_uncertainty_weighted", "k_center_greedy_uncertainty") and model_for_uncertainty is not None:
                new_idx = self.kcenter_uncertainty_weighted(model_for_uncertainty, k=k)
            elif self.strategy == "adaptive_homog_uwe" and model_for_uncertainty is not None:
                new_idx = self.kcenter_uncertainty_weighted(model_for_uncertainty, k=k)
            elif self.strategy == "dpp" and model_for_uncertainty is not None:
                new_idx = self.dpp(model_for_uncertainty, k=k)
            elif self.strategy == "weighted_tau_dpp" and model_for_uncertainty is not None:
                new_idx = self.weighted_tau_dpp(model_for_uncertainty, k=k)
            elif self.strategy == "weighted_dpp_tau1" and model_for_uncertainty is not None:
                new_idx = self.weighted_dpp_tau1(model_for_uncertainty, k=k)
            elif self.strategy == "tau_weighted_embedding" and model_for_uncertainty is not None:
                new_idx = self.tau_weighted_embedding(model_for_uncertainty, k=k)
            else:
                raise ValueError("strategy requires model_for_uncertainty")
            new_idx = self._ensure_preseed(new_idx)
        self.promote(new_idx)
        stem = f"{self.job_name}_cycle{self.cycle}"
        out_path = self._idx_path(stem)
        os.makedirs(self.idx_root, exist_ok=True)
        np.save(out_path, self.train_current_idx.astype(int))
        log_strategy_action(f"Saved updated indices to {out_path}.")
        return stem
