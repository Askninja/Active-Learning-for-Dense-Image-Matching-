
import os, os.path as osp, numpy as np, torch, cv2, time
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from roma.utils import get_tuple_transform_ops
from roma.datasets import OpticalMap
from roma.strategies.strategy_utils import (
    log_strategy_action,
    mean_entropy_score,
    compute_geometry_descriptor,
    geometry_novelty_scores,
    normalize_weights,
    k_center_greedy,
)
import roma.strategies.strategy_random               as _s_random
import roma.strategies.strategy_coreset              as _s_coreset
import roma.strategies.strategy_entropy              as _s_entropy
import roma.strategies.strategy_hs_cert              as _s_hs_cert
import roma.strategies.strategy_entropy_weighted_coreset   as _s_entropy_wc
import roma.strategies.strategy_hs_cert_weighted_coreset   as _s_hs_cert_wc
import roma.strategies.strategy_geometry_diversity   as _s_geometry
import roma.strategies.strategy_coreset_appearance   as _s_coreset_app
import roma.strategies.strategy_eigenvalue_diversity as _s_eigen
import roma.strategies.strategy_displacement_diversity as _s_disp
import roma.strategies.strategy_combined_eigen_displacement as _s_combined
import roma.strategies.strategy_hs_cert_weighted_eigenvalue_diversity as _s_hs_eigen
import matplotlib
matplotlib.use("Agg")


class ActiveLearningStrategy:
    def __init__(self, args, cycle: int, data_root, split: str, idx_root=None, rng_seed: int = 784) -> None:
        self.data_root = data_root
        self.idx_root = idx_root or osp.join(self.data_root, "Idx_files")
        self.job_name = getattr(args, "job_name", "run")
        self.strategy = getattr(args, "strategy", "random")
        self.train_resolution = getattr(args, "train_resolution", "low")
        self.temperature = float(getattr(args, "entropy_temperature", 0.5))
        self.selector_batch_size = int(getattr(args, "selector_batch_size", 8))
        self.geometry_hist_bins = int(getattr(args, "geometry_hist_bins", 16))
        self.geometry_conf_threshold = float(getattr(args, "geometry_conf_threshold", 0.5))
        self.geometry_chunk_size = int(getattr(args, "geometry_chunk_size", 2048))
        self.cycle = int(cycle)
        self.split = split
        self.train_pool_idx = self._load_idx(split)
        self.preseed_idx = self._load_idx("preseed_idx", required=False)
        self.train_current_idx = np.empty(0, dtype=int)
        if self.cycle == 0:
            if self.preseed_idx.size > 0:
                self.train_current_idx = np.unique(self.preseed_idx.astype(int))
        elif self.cycle > 0:
            prev_stem = f"{self.job_name}_cycle{self.cycle-1}"
            prev_path = self._idx_path(prev_stem)
            if osp.exists(prev_path):
                self.train_current_idx = np.load(prev_path).astype(int)
        self.rng = np.random.default_rng(int(rng_seed) + self.cycle)
        log_strategy_action(
            f"{self.job_name} cycle {self.cycle}: strategy={self.strategy}, "
            f"pool={self.train_pool_idx.size}, preseed={self.preseed_idx.size}, "
            f"current={self.train_current_idx.size}"
        )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _idx_path(self, name: str) -> str:
        fname = name if name.endswith(".npy") else f"{name}.npy"
        return osp.join(self.idx_root, fname)

    def _load_idx(self, name: str, required: bool = True) -> np.ndarray:
        path = self._idx_path(name)
        if not osp.exists(path):
            if required:
                raise FileNotFoundError(path)
            return np.empty(0, dtype=int)
        return np.load(path).astype(int)

    def _budget_for_cycle(self) -> int:
        schedule = {0: 10, 1: 20, 2: 20}
        pct = float(schedule.get(self.cycle, 30)) if self.cycle >= 0 else 0.0
        return int(np.round(pct / 100.0 * self.train_pool_idx.size))

    def remaining(self) -> np.ndarray:
        return np.setdiff1d(self.train_pool_idx, self.train_current_idx, assume_unique=True)

    def promote(self, new_idx: np.ndarray) -> None:
        new_idx = np.asarray(new_idx, dtype=int)
        if new_idx.size == 0:
            return
        self.train_current_idx = np.unique(np.concatenate([self.train_current_idx, new_idx]))

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _entropy_dataset(self, idx_subset: np.ndarray):
        resolutions = {
            "low": (448, 448),
            "medium": (14 * 8 * 5, 14 * 8 * 5),
            "high": (14 * 8 * 6, 14 * 8 * 6),
        }
        ht, wt = resolutions[self.train_resolution]
        dataset = OpticalMap(
            data_root=self.data_root,
            ht=ht, wt=wt,
            use_horizontal_flip_aug=False,
            use_cropping_aug=False,
            use_color_jitter_aug=False,
            use_swap_aug=False,
            use_dual_cropping_aug=False,
            split=f"Idx_files/{self.split}",
        )
        dataset.train_idx = np.asarray(idx_subset, dtype=int)
        return dataset

    def _select_gm_cls_scale(self, corresps):
        gm_scales = [s for s, p in corresps.items() if p.get("gm_cls") is not None]
        if not gm_scales:
            raise ValueError("RoMa forward pass did not return any coarse gm_cls outputs")
        return max(gm_scales)

    def _select_flow_scale(self, corresps):
        flow_scales = [s for s, p in corresps.items() if p.get("flow") is not None]
        if not flow_scales:
            raise ValueError("RoMa forward pass did not return any flow outputs")
        return max(flow_scales)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _pair_embedding_fine(self, model, batch) -> np.ndarray:
        feature_pyramid = model.extract_backbone_features(batch, batched=True, upsample=False)
        finest_scale = min(feature_pyramid.keys())
        features = feature_pyramid[finest_scale]
        feat_a, feat_b = features.chunk(2, dim=0)
        embedding = torch.cat((feat_a.mean(dim=(2, 3)), feat_b.mean(dim=(2, 3))), dim=1)
        embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
        return embedding[0].detach().float().cpu().numpy()

    def _compute_fine_feature_embeddings(self, model, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dataset = self._entropy_dataset(avail)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        device = next(model.parameters()).device
        embeddings = []
        model.eval()
        with torch.no_grad():
            for _, batch in zip(avail.astype(int), dataloader):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                embeddings.append(self._pair_embedding_fine(model, batch))
        if not embeddings:
            return np.empty(0, dtype=int), np.empty((0, 0), dtype=np.float32)
        return avail.astype(int), np.asarray(embeddings, dtype=np.float32)

    def _compute_geometry_descriptors(self, model, idx_subset: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        idx_subset = np.asarray(idx_subset, dtype=int)
        if idx_subset.size == 0:
            return np.empty(0, dtype=int), torch.empty((0, 2 * self.geometry_hist_bins + 1))
        dataset = self._entropy_dataset(idx_subset)
        dataloader = DataLoader(
            dataset, batch_size=max(1, self.selector_batch_size),
            shuffle=False, num_workers=0, pin_memory=True,
        )
        device = next(model.parameters()).device
        descriptors = []
        flow_scale = None
        consumed = 0
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_size = int(batch["im_A"].shape[0])
                batch_indices = idx_subset[consumed:consumed + batch_size]
                consumed += batch_size
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                corresps = model(batch)
                if flow_scale is None:
                    flow_scale = self._select_flow_scale(corresps)
                flow = corresps[flow_scale]["flow"]
                confidence = torch.sigmoid(corresps[flow_scale]["certainty"])
                descriptor = compute_geometry_descriptor(
                    flow, confidence,
                    bins=self.geometry_hist_bins,
                    conf_threshold=self.geometry_conf_threshold,
                )
                descriptors.append((batch_indices.astype(int), descriptor.detach().cpu()))
        if not descriptors:
            return np.empty(0, dtype=int), torch.empty((0, 2 * self.geometry_hist_bins + 1))
        sample_ids = np.concatenate([s for s, _ in descriptors]).astype(int)
        descriptor_tensor = torch.cat([d for _, d in descriptors], dim=0)
        return sample_ids, descriptor_tensor

    # ------------------------------------------------------------------
    # Uncertainty scoring
    # ------------------------------------------------------------------

    def _get_matches(self, flow, certainty, H, W, num_matches=5000):
        x_coords = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=flow.device)
        y_coords = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=flow.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords_A = torch.stack((xx, yy), dim=-1)[None]
        coords_B = coords_A + flow.permute(0, 2, 3, 1)
        matches = torch.cat((coords_A, coords_B), dim=-1).reshape(1, -1, 4)[0]
        cert = torch.nan_to_num(certainty.sigmoid().reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0)
        if cert.sum() <= 0:
            cert = torch.ones_like(cert)
        num_matches = min(num_matches, len(cert))
        if num_matches == 0:
            return matches.new_zeros((0, 4)).cpu().numpy()
        return matches[torch.multinomial(cert, num_matches, replacement=False)].cpu().numpy()

    def _compute_hs_uncertainty(self, M, H, W, K=10, P=50):
        M_pixel = M * np.array([W/2, H/2, W/2, H/2]) + np.array([W/2, H/2, W/2, H/2])
        OA_pixel = np.random.rand(P, 2) * np.array([W, H])
        OA_norm = (OA_pixel - np.array([W/2, H/2])) / np.array([W/2, H/2])
        projections = []
        for _ in range(K):
            subset_size = min(1000, len(M))
            idx = np.random.choice(len(M), subset_size, replace=False)
            M_k = M[idx] * np.array([W/2, H/2, W/2, H/2]) + np.array([W/2, H/2, W/2, H/2])
            H_mat, _ = cv2.findHomography(M_k[:, :2], M_k[:, 2:], cv2.RANSAC, 5.0)
            if H_mat is None:
                H_mat = np.eye(3)
            OA_hom = np.hstack((OA_norm, np.ones((P, 1))))
            proj_hom = OA_hom @ H_mat.T
            projections.append(proj_hom[:, :2] / proj_hom[:, 2:3])
        projections = np.array(projections)
        stds = [np.sqrt(np.std(projections[:, i, 0])**2 + np.std(projections[:, i, 1])**2) for i in range(P)]
        return 1 - 1 / (1 + np.mean(stds))

    def _score_avail(self, model, avail: np.ndarray, score_name: str) -> tuple[np.ndarray, np.ndarray]:
        if avail.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
        dataset = self._entropy_dataset(avail)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        device = next(model.parameters()).device
        model.eval()
        scores = []
        gm_scale = finest_scale = None
        with torch.no_grad():
            for sample_idx, batch in zip(avail.astype(int), dataloader):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                corresps = model(batch)
                if score_name == "entropy":
                    if gm_scale is None:
                        gm_scale = self._select_gm_cls_scale(corresps)
                    gm_cls = corresps[gm_scale]["gm_cls"].detach().float().cpu().numpy()
                    score = float(mean_entropy_score(gm_cls, temperature=self.temperature))
                elif score_name == "hs_cert":
                    if finest_scale is None:
                        finest_scale = max(corresps.keys())
                    flow = corresps[finest_scale]["flow"]
                    certainty = corresps[finest_scale]["certainty"]
                    H, W = flow.shape[-2:]
                    M = self._get_matches(flow, certainty, H, W, num_matches=5000)
                    score = float(self._compute_hs_uncertainty(M, H, W, K=10, P=50))
                else:
                    raise ValueError(f"Unknown score_name '{score_name}'")
                scores.append((int(sample_idx), score))
        sample_ids = np.asarray([i for i, _ in scores], dtype=int)
        values = np.asarray([s for _, s in scores], dtype=np.float32)
        return sample_ids, values

    # ------------------------------------------------------------------
    # RANSAC homographies
    # ------------------------------------------------------------------

    def _compute_ransac_homographies(
        self, model, avail: np.ndarray, K: int = 50, num_matches: int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
        if avail.size == 0:
            return np.empty(0, dtype=int), np.empty((0, K, 3, 3), dtype=np.float64)
        dataset = self._entropy_dataset(avail)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        device = next(model.parameters()).device
        model.eval()
        all_sample_ids, all_homographies = [], []
        selected_scale = None
        subset_size = 500
        with torch.no_grad():
            for sample_idx, batch in zip(avail.astype(int), dataloader):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                corresps = model(batch)
                if selected_scale is None:
                    selected_scale = max(corresps.keys())
                flow = corresps[selected_scale]["flow"]
                certainty = corresps[selected_scale]["certainty"]
                H_img, W_img = flow.shape[-2:]
                M = self._get_matches(flow, certainty, H_img, W_img, num_matches=num_matches)
                M_px = M * np.array([W_img/2, H_img/2, W_img/2, H_img/2]) \
                         + np.array([W_img/2, H_img/2, W_img/2, H_img/2])
                kpts_A, kpts_B = M_px[:, :2], M_px[:, 2:]
                n_pts = len(M)
                ss = min(subset_size, n_pts)
                homographies_k = np.zeros((K, 3, 3), dtype=np.float64)
                for k_iter in range(K):
                    if ss < 4:
                        homographies_k[k_iter] = np.eye(3)
                        continue
                    idx = np.random.choice(n_pts, ss, replace=False)
                    H_mat, _ = cv2.findHomography(kpts_A[idx], kpts_B[idx], cv2.RANSAC, 5.0)
                    homographies_k[k_iter] = H_mat if H_mat is not None else np.eye(3)
                all_sample_ids.append(int(sample_idx))
                all_homographies.append(homographies_k)
        if not all_sample_ids:
            return np.empty(0, dtype=int), np.empty((0, K, 3, 3), dtype=np.float64)
        return np.asarray(all_sample_ids, dtype=int), np.stack(all_homographies, axis=0)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def get_train_idx(self, model_for_uncertainty=None) -> str:
        k = self._budget_for_cycle()

        if self.strategy == "preseed":
            new_idx = self.train_current_idx if self.cycle != 0 else self.preseed_idx.astype(int)
            if self.cycle == 0 and self.preseed_idx.size == 0:
                raise ValueError("preseed_idx.npy missing")

        elif self.strategy == "full":
            new_idx = self.train_current_idx if self.cycle != 0 else \
                np.unique(np.concatenate([self.train_pool_idx, self.preseed_idx])).astype(int)

        elif self.strategy == "random":
            new_idx = _s_random.run(self, k)

        elif self.strategy == "coreset":
            new_idx = _s_coreset.run(self, k, model_for_uncertainty)

        elif self.strategy == "entropy":
            new_idx = _s_entropy.run(self, k, model_for_uncertainty)

        elif self.strategy == "hs_cert":
            new_idx = _s_hs_cert.run(self, k, model_for_uncertainty)

        elif self.strategy == "entropy_weighted_coreset":
            new_idx = _s_entropy_wc.run(self, k, model_for_uncertainty)

        elif self.strategy == "hs_cert_weighted_coreset":
            new_idx = _s_hs_cert_wc.run(self, k, model_for_uncertainty)

        elif self.strategy == "geometry_diversity":
            new_idx = _s_geometry.run(self, k, model_for_uncertainty)

        elif self.strategy == "coreset_appearance":
            new_idx = _s_coreset_app.run(self, k, model_for_uncertainty)

        elif self.strategy == "eigenvalue_diversity":
            new_idx = _s_eigen.run(self, k, model_for_uncertainty)

        elif self.strategy == "displacement_diversity":
            new_idx = _s_disp.run(self, k, model_for_uncertainty)

        elif self.strategy == "combined_eigen_displacement":
            new_idx = _s_combined.run(self, k, model_for_uncertainty)

        elif self.strategy == "hs_cert_weighted_eigenvalue_diversity":
            new_idx = _s_hs_eigen.run(self, k, model_for_uncertainty)

        else:
            raise ValueError(f"Strategy '{self.strategy}' is not implemented.")

        self.promote(new_idx)
        stem = f"{self.job_name}_cycle{self.cycle}"
        out_path = self._idx_path(stem)
        os.makedirs(self.idx_root, exist_ok=True)
        np.save(out_path, self.train_current_idx.astype(int))
        log_strategy_action(f"Saved updated indices to {out_path}.")
        return stem
