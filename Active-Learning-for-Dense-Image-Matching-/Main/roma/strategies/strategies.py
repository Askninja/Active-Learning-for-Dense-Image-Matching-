
import os, os.path as osp, numpy as np, torch, cv2, time
import inspect
import math
from PIL import Image
from roma.utils import get_tuple_transform_ops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def log_strategy_action(message: str):
    print(f"[STRATEGY] {message}", flush=True)


class ActiveLearningStrategy:
    def __init__(self, args, cycle: int, data_root, split: str, idx_root=None, rng_seed: int = 784) -> None:
        self.data_root = data_root
        self.idx_root = idx_root or osp.join(self.data_root, "Idx_files")
        self.job_name = getattr(args, "job_name", "run")
        self.split = split
        self.strategy = getattr(args, "strategy", "coreset")
        custom_tsne_dir = getattr(args, "tsne_plot_dir", None)
        if custom_tsne_dir:
            custom_tsne_dir = osp.expanduser(custom_tsne_dir)
        self.tsne_plot_dir = custom_tsne_dir
        self._tsne_warned = False
        self.cycle = int(cycle)
        self.train_pool_idx = self._load_idx(split)
        self.preseed_idx = self._load_idx("preseed_idx", required=False)
        self.hs_cert_thresh = getattr(args, "hs_cert_thresh", None)
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

    def _save_hs_cert_plot(
        self,
        values: np.ndarray,
        mu: float,
        sigma: float,
        tau: float,
        stem: str,
        out_dir: str | None = None,
        value_label: str = "hs_cert",
    ) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=float)
        out_dir = out_dir or osp.dirname(__file__)
        os.makedirs(out_dir, exist_ok=True)
        suffix = value_label if value_label else "hs_cert"
        fname = f"{stem}_{suffix}.png" if stem else f"{suffix}.png"
        out_path = osp.join(out_dir, fname)
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=40, range=(0.0, 1.0), color="steelblue", alpha=0.7)
        plt.axvline(mu, color="orange", linestyle="--", linewidth=2, label=f"mu={mu:.3f}")
        plt.xlabel(value_label or "value")
        plt.ylabel("count")
        title_label = value_label or "value"
        plt.title(f"{self.job_name} cycle {self.cycle} {title_label} stats")
        info = f"mu={mu:.4f}\nsigma={sigma:.4f}\ntau={tau:.4f}"
        plt.text(0.98, 0.95, info, ha="right", va="top", transform=plt.gca().transAxes, fontsize=11,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        log_strategy_action(f"Saved hs_cert plot to {out_path}.")

    def _save_tsne_plot(
        self,
        embeddings: torch.Tensor,
        values,
        picked_local,
        stem: str,
        value_label: str = "value",
        out_dir: str | None = None,
        max_points: int = 2000,
    ) -> None:
        if embeddings is None or embeddings.numel() == 0:
            return
        if embeddings.dim() != 2 or embeddings.shape[0] < 3:
            return
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            if not self._tsne_warned:
                log_strategy_action("scikit-learn not available; skipping t-SNE plots.")
                self._tsne_warned = True
            return
        embs = embeddings.detach().float().cpu().numpy()
        N = embs.shape[0]
        if N < 3:
            return
        picked_local = np.asarray(picked_local, dtype=int) if picked_local is not None else np.empty(0, dtype=int)
        picked_local = picked_local[(picked_local >= 0) & (picked_local < N)]
        picked_local = np.unique(picked_local)
        picked_mask = np.zeros(N, dtype=bool)
        if picked_local.size > 0:
            picked_mask[picked_local] = True
        keep_idx = np.arange(N)
        max_points = int(max(64, max_points))
        if N > max_points:
            rng = np.random.default_rng(self.rng_seed + self.cycle + 1337)
            keepers = np.where(picked_mask)[0]
            budget = max_points - keepers.size
            if budget > 0:
                remaining = np.where(~picked_mask)[0]
                if remaining.size > budget:
                    sampled = rng.choice(remaining, size=budget, replace=False)
                else:
                    sampled = remaining
                keep_idx = np.concatenate([keepers, sampled])
            else:
                keep_idx = keepers
            keep_idx = np.unique(keep_idx)
        embs_sub = embs[keep_idx]
        if embs_sub.shape[0] < 3:
            return
        picked_mask_sub = picked_mask[keep_idx]
        values_arr = None
        if values is not None:
            values_arr = np.asarray(values, dtype=float).reshape(-1)
            if values_arr.shape[0] != N:
                log_strategy_action(
                    f"Skipping value-driven coloring for t-SNE (expected {N} values, got {values_arr.shape[0]})."
                )
                values_arr = None
        if values_arr is not None:
            values_arr = values_arr[keep_idx]
            if not np.all(np.isfinite(values_arr)):
                finite = np.isfinite(values_arr)
                if finite.any():
                    median_val = float(np.median(values_arr[finite]))
                else:
                    median_val = 0.0
                values_arr = np.where(finite, values_arr, median_val)
            v_min = float(values_arr.min())
            v_max = float(values_arr.max())
            if abs(v_max - v_min) > 1e-12:
                values_norm = (values_arr - v_min) / (v_max - v_min)
            else:
                values_norm = np.zeros_like(values_arr)
        else:
            values_norm = None
        perplexity = min(30, max(5, embs_sub.shape[0] // 3))
        perplexity = min(perplexity, embs_sub.shape[0] - 1)
        perplexity = max(2, perplexity)
        if perplexity >= embs_sub.shape[0]:
            perplexity = embs_sub.shape[0] - 1
        if perplexity < 1:
            return
        tsne_params = inspect.signature(TSNE.__init__).parameters
        tsne_kwargs = {
            "n_components": 2,
            "init": "pca",
            "random_state": self.rng_seed + self.cycle,
            "perplexity": perplexity,
            "metric": "cosine",
        }
        if "learning_rate" in tsne_params:
            tsne_kwargs["learning_rate"] = "auto"
        if "n_iter" in tsne_params:
            tsne_kwargs["n_iter"] = 1500
        try:
            tsne = TSNE(**tsne_kwargs)
            coords = tsne.fit_transform(embs_sub)
        except Exception as exc:
            log_strategy_action(f"t-SNE failed ({exc}); falling back to PCA scatter.")
            try:
                from sklearn.decomposition import PCA

                coords = PCA(n_components=2).fit_transform(embs_sub)
            except Exception as pca_exc:
                log_strategy_action(f"PCA fallback failed: {pca_exc}")
                return
        out_dir = out_dir or self.tsne_plot_dir
        if not out_dir:
            out_dir = osp.join(osp.dirname(__file__), "tsne_plots")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{stem}_tsne.png" if stem else "tsne.png"
        out_path = osp.join(out_dir, fname)
        fig, ax = plt.subplots(figsize=(6, 5))
        if values_norm is not None:
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values_norm,
                cmap="plasma",
                s=18,
                alpha=0.75,
                linewidths=0,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(value_label)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color="steelblue", s=18, alpha=0.75, linewidths=0)
        if picked_mask_sub.any():
            ax.scatter(
                coords[picked_mask_sub, 0],
                coords[picked_mask_sub, 1],
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.2,
                s=60,
                label="selected",
            )
            ax.legend(loc="best")
        ax.set_xlabel("tsne-1")
        ax.set_ylabel("tsne-2")
        ax.set_title(f"{self.job_name} cycle {self.cycle} t-SNE")
        fig.tight_layout()
        fig.savefig(out_path, dpi=175)
        plt.close(fig)
        log_strategy_action(f"Saved t-SNE plot to {out_path}.")

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
        schedule = {0: 10, 1: 20, 2: 25}
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

    def _hs_cert_scores(self, model_for_uncertainty, avail: np.ndarray) -> np.ndarray:
        """Compute homography-stability based certainty for each candidate pair."""
        t0 = time.time()
        subset_cap = 2000
        ransac_iters = 50
        hs_vals = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            dense_matches, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            sparse_matches, _ = model_for_uncertainty.sample(
                dense_matches, dense_certainty, 5000, thresh_score=0.05, sample_seed=int(i)
            )
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
            subset = min(subset_cap, A_px.shape[0])
            thresh = 3 * min(w2, h2) / 480
            for _ in range(ransac_iters):
                sel = (
                    g.choice(A_px.shape[0], size=subset, replace=False)
                    if A_px.shape[0] > subset
                    else np.arange(A_px.shape[0])
                )
                pA, pB = A_px[sel], B_px[sel]
                H, _ = cv2.findHomography(
                    pA, pB, method=cv2.RANSAC, ransacReprojThreshold=thresh, confidence=0.999
                )
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
        self.last_hs_cert_log = {
            "hs_cert_seconds": float(time.time() - t0),
            "hs_cert_pool_size": int(avail.size),
            "hs_cert_subset_cap": int(subset_cap),
            "hs_cert_ransac_iters": int(ransac_iters),
        }
        return hs_cert

    @torch.no_grad()
    def _mean_uncertainty_scores(self, model_for_uncertainty, avail: np.ndarray) -> np.ndarray:
        """Compute simple uncertainty as 1 - mean certainty over the dense certainty map."""
        uncerts = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            _, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            try:
                mean_cert = float(dense_certainty.mean().item())
            except Exception:
                mean_cert = float(torch.as_tensor(dense_certainty).mean().item())
            mean_cert = float(np.clip(mean_cert, 0.0, 1.0))
            uncerts.append(1.0 - mean_cert)
        return np.asarray(uncerts, dtype=float)

    @torch.no_grad()
    def _entropy_scores(self, model_for_uncertainty, avail: np.ndarray) -> np.ndarray:
        """Compute image-level entropy from the dense certainty map."""
        scores = []
        for i in avail:
            a_path, b_path = self._idx_to_paths(int(i))
            _, dense_certainty = model_for_uncertainty.match(a_path, b_path)
            p = torch.as_tensor(dense_certainty).float().clamp(1e-6, 1.0 - 1e-6)
            ent = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
            scores.append(float(ent.mean().item()))
        return np.asarray(scores, dtype=float)

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
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
        s_min, s_max = float(hs_cert.min()), float(hs_cert.max())
        hs_cert_norm = (hs_cert - s_min) / (s_max - s_min + 1e-8)
        mu = float(hs_cert_norm.mean())
        sigma = float(hs_cert_norm.std())
        tau = mu + sigma
        plot_dir = osp.join(self.data_root, "weighted_tau_dpp_plots")
        self._save_hs_cert_plot(
            hs_cert_norm,
            mu,
            sigma,
            tau,
            f"{self.job_name}_cycle{self.cycle}_weighted_tau",
            out_dir=plot_dir,
        )
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
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
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
            sparse_matches, _ = model_for_uncertainty.sample(
                dense_matches,
                dense_certainty,
                5000,
                thresh_score=0.05,
                sample_seed=int(idx),
            )
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
        if k == 0:
            return np.empty(0, dtype=int)
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
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail).astype(np.float64)
        mu = float(hs_cert.mean())
        sigma = float(hs_cert.std())
        X = hs_cert.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=getattr(self, "rng_seed", 0))
        gmm.fit(X)
        means = gmm.means_.flatten()
        covs = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()
        idx = np.argsort(means)
        m1, m2 = means[idx]
        v1, v2 = covs[idx]
        w1, w2 = weights[idx]
        sep = abs(m2 - m1) / math.sqrt(0.5 * (v1 + v2))
        bim_sep = sep * (4.0 * w1 * w2)
        frac_low = float((hs_cert <= 0.2).mean())
        tau = mu * bim_sep * (1.0 - frac_low)
        score_global = tau
        stem = f"{self.job_name}_cycle{self.cycle}_kcenter"
        self._save_hs_cert_plot(hs_cert, mu, sigma, tau, stem)
        u_norm = 1.0 - hs_cert
        scale = (u_norm ** tau)
        scale = scale.astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        self._save_tsne_plot(
            scaled_embs,
            values=hs_cert,
            picked_local=picked_local,
            stem=f"{stem}_weighted",
            value_label="hs_cert",
        )
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def kcenter_uncertainty_weighted_raw(self, model_for_uncertainty, k: int) -> np.ndarray:
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
        cand_embs = torch.stack(
            [self._pair_embedding_raw(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        seed_embs = self._seed_embeddings_raw(encoder, device, tform)
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail).astype(np.float64)
        mu = float(hs_cert.mean())
        sigma = float(hs_cert.std())
        if self.hs_cert_thresh is None:
            a_thresh = 0.2
            b_thresh = 0.8
        else:
            user_thresh = max(0.0, min(1.0, float(self.hs_cert_thresh)))
            # Use symmetric thresholds around 0.5 (e.g., 0.2 -> 0.2/0.8, 0.25 -> 0.25/0.75).
            a_thresh = min(user_thresh, 1.0 - user_thresh)
            b_thresh = max(user_thresh, 1.0 - user_thresh)
        eps = 1e-6
        pL = float((hs_cert < a_thresh).mean())
        pR = float((hs_cert > b_thresh).mean())
        pE = pL + pR
        balance = 1.0 - (abs(pL - pR) / (pE + eps))
        tau = pE * balance + mu
        stem = f"{self.job_name}_cycle{self.cycle}_kcenter_raw"
        self._save_hs_cert_plot(hs_cert, mu, sigma, tau, stem, value_label="hs_cert")
        uncertainty = np.clip(1.0 - hs_cert, 0.0, 1.0)
        scale = (uncertainty ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        self._save_tsne_plot(
            scaled_embs,
            values=hs_cert,
            picked_local=picked_local,
            stem=f"{stem}_weighted",
            value_label="hs_cert",
        )
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def kcenter_entropy_weighted_raw(self, model_for_uncertainty, k: int) -> np.ndarray:
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
        cand_embs = torch.stack(
            [self._pair_embedding_raw(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        seed_embs = self._seed_embeddings_raw(encoder, device, tform)
        entropy = self._entropy_scores(model_for_uncertainty, avail).astype(np.float64)
        max_ent = float(math.log(2.0))
        entropy_norm = np.clip(entropy / max_ent, 0.0, 1.0)
        mu = float(entropy_norm.mean())
        sigma = float(entropy_norm.std())
        a_thresh = 0.2
        b_thresh = 0.8
        eps = 1e-6
        pL = float((entropy_norm < a_thresh).mean())
        pR = float((entropy_norm > b_thresh).mean())
        pE = pL + pR
        balance = 1.0 - (abs(pL - pR) / (pE + eps))
        tau = pE * balance + mu
        stem = f"{self.job_name}_cycle{self.cycle}_kcenter_entropy_raw"
        self._save_hs_cert_plot(entropy_norm, mu, sigma, tau, stem, value_label="entropy")
        uncertainty = entropy_norm
        scale = (uncertainty ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        self._save_tsne_plot(
            scaled_embs,
            values=entropy_norm,
            picked_local=picked_local,
            stem=f"{stem}_weighted",
            value_label="entropy",
        )
        picked = avail[picked_local]
        return picked.astype(int)



    @torch.no_grad()
    def kcenter_uncertainty_embedding(self, model_for_uncertainty, k: int) -> np.ndarray:
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
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        seed_embs = self._seed_embeddings(encoder, device, tform)
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
        uncertainty = np.clip(1.0 - hs_cert, 0.0, 1.0).astype(np.float32)
        scale_t = torch.from_numpy(uncertainty).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        self._save_tsne_plot(
            scaled_embs,
            values=hs_cert,
            picked_local=picked_local,
            stem=f"{self.job_name}_cycle{self.cycle}_embedding",
            value_label="hs_cert",
        )
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
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
        hs_cert_norm = hs_cert
        mu = float(hs_cert_norm.mean())
        sigma = float(hs_cert_norm.std())
        tau = mu + sigma
        stem = f"{self.job_name}_cycle{self.cycle}"
        self._save_hs_cert_plot(hs_cert_norm, mu, sigma, tau, stem)
        scale = (hs_cert_norm ** tau).astype(np.float32)
        scale_t = torch.from_numpy(scale).view(-1, 1)
        scaled_embs = cand_embs * scale_t
        picked_local = self._kcenter_from_vecs(scaled_embs, k=k, seed_X=seed_embs)
        self._save_tsne_plot(
            scaled_embs,
            values=hs_cert_norm,
            picked_local=picked_local,
            stem=f"{stem}_tau_weighted",
            value_label="hs_cert",
        )
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def kmeans_diversity(self, model_for_uncertainty, k: int) -> np.ndarray:
        """Select k samples by clustering candidate embeddings and picking points closest to centroids."""
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
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        
        emb_np = cand_embs.cpu().numpy()
        n_clusters = max(1, min(int(k), emb_np.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.rng_seed + self.cycle, n_init=10)
        kmeans.fit(emb_np)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_[labels]
        dis = ((emb_np - centers) ** 2).sum(axis=1)
        picked_local = []
        for c in range(n_clusters):
            mask = labels == c
            if not np.any(mask):
                continue
            local_idx = np.where(mask)[0]
            best_local = local_idx[dis[mask].argmin()]
            picked_local.append(best_local)
        if not picked_local:
            return np.empty(0, dtype=int)
        picked_local = np.array(picked_local, dtype=int)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def clue(self, model_for_uncertainty, k: int) -> np.ndarray:
        """CLUE-style selection: cluster uncertainty-scaled embeddings and pick the most uncertain per cluster."""
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
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail).astype(np.float64)
        mu = float(hs_cert.mean())
        sigma = float(hs_cert.std())
        x = np.sort(hs_cert)
        N = x.shape[0]
        F = (np.arange(1, N + 1, dtype=np.float64)) / (N + 1.0)
        q = 0.2
        k_left = int(np.floor(N * q))
        if k_left < 3:
            k_left = 3
        if k_left > N - 3:
            k_left = N - 3
        x_left = x[:k_left]
        F_left = F[:k_left]
        y_left = np.log(F_left)
        aL, bL, cL = np.polyfit(x_left, y_left, 2)
        LTC_L = 2.0 * aL
        k_right = int(np.ceil(N * (1.0 - q)))
        if k_right < 0:
            k_right = 0
        if k_right > N - 3:
            k_right = N - 3
        x_right = x[k_right:]
        F_right = F[k_right:]
        y_right = np.log(1.0 - F_right)
        aR, bR, cR = np.polyfit(x_right, y_right, 2)
        LTC_R = 2.0 * aR
        S_L = abs(LTC_L)
        S_R = abs(LTC_R)
        denom = S_L + S_R
        B_low = S_L / denom
        B_high = S_R / denom
        tau = 2.0 * (1.0 - abs(B_high - B_low) + B_high)
        uncertainty = np.clip(1.0 - hs_cert, 0.0, 1.0)
        uncertainty_tau = (uncertainty ** tau).astype(np.float32)
        scaled = cand_embs * torch.from_numpy(uncertainty_tau).view(-1, 1)
        scaled_np = scaled.cpu().numpy()
        n_clusters = max(1, min(int(k), scaled_np.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.rng_seed + self.cycle, n_init=10)
        kmeans.fit(scaled_np)
        labels = kmeans.labels_
        picked_local = []
        for c in range(n_clusters):
            mask = labels == c
            if not np.any(mask):
                continue
            cluster_idx = np.where(mask)[0]
            cluster_unc = uncertainty_tau[cluster_idx]
            best_local = cluster_idx[int(cluster_unc.argmax())]
            picked_local.append(best_local)
        if not picked_local:
            return np.empty(0, dtype=int)
        picked_local = np.array(picked_local, dtype=int)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def badge(self, model_for_uncertainty, k: int) -> np.ndarray:
        """BADGE-style selection using hs_cert-derived pseudo-probabilities and k-means++ over gradient embeddings."""
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
        cand_embs = torch.stack(
            [self._pair_embedding_scale1(encoder, device, tform, *self._idx_to_paths(int(i))) for i in avail],
            dim=0,
        ).float()
        cand_embs = cand_embs / (cand_embs.norm(dim=1, keepdim=True) + 1e-8)
        emb_np = cand_embs.cpu().numpy()
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
        # treat hs_cert as prob for class 1 in a 2-class view of certainty
        probs = np.stack([1.0 - hs_cert, hs_cert], axis=1)
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        y_hat = np.argmax(probs, axis=1)
        p_shift = -probs
        p_shift[np.arange(probs.shape[0]), y_hat] += 1.0
        # gradient embedding per BADGE: outer product of shifted prob with feature vector
        grad_embs = np.concatenate([p_shift[:, [c]] * emb_np for c in range(probs.shape[1])], axis=1)
        n_clusters = max(1, min(int(k), grad_embs.shape[0]))
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.rng_seed + self.cycle, n_init=10)
        kmeans.fit(grad_embs)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_[labels]
        dis = ((grad_embs - centers) ** 2).sum(axis=1)
        picked_local = []
        for c in range(n_clusters):
            mask = labels == c
            if not np.any(mask):
                continue
            local_idx = np.where(mask)[0]
            best_local = local_idx[dis[mask].argmin()]
            picked_local.append(best_local)
        if not picked_local:
            return np.empty(0, dtype=int)
        picked_local = np.array(picked_local, dtype=int)
        picked = avail[picked_local]
        return picked.astype(int)

    @torch.no_grad()
    def normal_uncertainty(self, model_for_uncertainty, k: int) -> np.ndarray:
        """Simple uncertainty: rank by 1 - mean certainty over dense certainty map."""
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        uncertainty = self._mean_uncertainty_scores(model_for_uncertainty, avail)
        order = np.argsort(-uncertainty)
        picked = avail[order[:k]]
        return picked.astype(int)

    @torch.no_grad()
    def entropy(self, model_for_uncertainty, k: int) -> np.ndarray:
        """Select images with highest entropy over dense certainty map."""
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        entropy = self._entropy_scores(model_for_uncertainty, avail)
        order = np.argsort(-entropy)
        picked = avail[order[:k]]
        return picked.astype(int)

    @torch.no_grad()
    def uncertainty(self, model_for_uncertainty, k: int) -> np.ndarray:
        avail = self.remaining()
        k = min(int(k), avail.size)
        if k == 0:
            return np.empty(0, dtype=int)
        hs_cert = self._hs_cert_scores(model_for_uncertainty, avail)
        order = np.argsort(hs_cert)
        picked = avail[order[:k]]
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
            elif self.strategy == "kcenter_uncertainty_weighted_raw" and model_for_uncertainty is not None:
                new_idx = self.kcenter_uncertainty_weighted_raw(model_for_uncertainty, k=k)
            elif self.strategy == "kcenter_entropy_weighted_raw" and model_for_uncertainty is not None:
                new_idx = self.kcenter_entropy_weighted_raw(model_for_uncertainty, k=k)
            elif self.strategy == "kcenter_uncertainty_embedding" and model_for_uncertainty is not None:
                new_idx = self.kcenter_uncertainty_embedding(model_for_uncertainty, k=k)
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
            elif self.strategy == "kmeans_diversity" and model_for_uncertainty is not None:
                new_idx = self.kmeans_diversity(model_for_uncertainty, k=k)
            elif self.strategy == "clue" and model_for_uncertainty is not None:
                new_idx = self.clue(model_for_uncertainty, k=k)
            elif self.strategy == "badge" and model_for_uncertainty is not None:
                new_idx = self.badge(model_for_uncertainty, k=k)
            elif self.strategy == "normal_uncertainty" and model_for_uncertainty is not None:
                new_idx = self.normal_uncertainty(model_for_uncertainty, k=k)
            elif self.strategy == "entropy" and model_for_uncertainty is not None:
                new_idx = self.entropy(model_for_uncertainty, k=k)
            elif self.strategy == "uncertainty" and model_for_uncertainty is not None:
                new_idx = self.uncertainty(model_for_uncertainty, k=k)
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
