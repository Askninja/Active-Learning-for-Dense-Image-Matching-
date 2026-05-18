from PIL import Image
import numpy as np
import os.path as osp
from tqdm import tqdm
import cv2
from roma.utils import pose_auc
from roma.utils.utils import angle_error_mat, angle_error_vec


class METUVisTIRFlatBenchmark:
    """Fundamental-matrix benchmark for the flattened METU VisTIR dataset.

    Expects the flat layout produced by make_metu_vistir_dataset.py:
        {data_root}/pair{N}_1.jpg     visible image
        {data_root}/pair{N}_2.jpg     thermal image
        {data_root}/gt_{N}.txt        3×3 fundamental matrix F
        {data_root}/ka_{N}.txt        3×3 camera intrinsics K_A (visible)
        {data_root}/kb_{N}.txt        3×3 camera intrinsics K_B (thermal)
        {data_root}/{split}.npy            1-based pair index array (split is the relative path, e.g. "Idx_files/train")

    Evaluation metric: pose AUC at 5° / 10° / 20°.
    Keys returned by benchmark() match the superset expected by al_utils.benchmark_metrics
    (auc_5, auc_10 are used; auc_3 / epe return None gracefully).
    """

    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        idx_path = osp.join(data_root, f"{split}.npy")
        self.test_idx = np.load(idx_path)
        print(f"METU_VisTIR [{split}]: {self.test_idx.size} pairs")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pixel(coords, w, h):
        offset = 0.5
        x = w * (coords[..., 0] + 1) / 2.0 - offset
        y = h * (coords[..., 1] + 1) / 2.0 - offset
        return np.stack([x, y], axis=-1)

    @staticmethod
    def _estimate_pose(pos_a, pos_b, K_A, K_B):
        """Normalise with respective K, estimate E via MAGSAC.

        Returns (R, t, pts_a_n, pts_b_n) or (None, None, None, None).
        The normalised points are returned so the caller can reuse them for
        the GT cheirality check without recomputing undistortPoints.
        """
        pts_a_n = cv2.undistortPoints(
            pos_a.reshape(-1, 1, 2).astype(np.float64), K_A, None
        ).reshape(-1, 2)
        pts_b_n = cv2.undistortPoints(
            pos_b.reshape(-1, 1, 2).astype(np.float64), K_B, None
        ).reshape(-1, 2)

        try:
            E, _ = cv2.findEssentialMat(
                pts_a_n, pts_b_n,
                cameraMatrix=np.eye(3),
                method=cv2.USAC_MAGSAC,
                prob=0.99999,
                threshold=1e-3,
            )
        except Exception:
            return None, None, None, None

        if E is None or E.shape != (3, 3):
            return None, None, None, None

        _, R, t, _ = cv2.recoverPose(E, pts_a_n, pts_b_n)
        return R, t, pts_a_n, pts_b_n

    @staticmethod
    def _gt_pose_from_F(F_gt, K_A, K_B):
        """Extract (R_gt, t_gt) using fixed synthetic 3D points for cheirality.

        Fully independent of model predictions — GT is stable regardless of match quality.
        Enumerates all 4 E decompositions and picks the one where the most synthetic
        points have positive depth in both cameras.
        """
        E_gt = K_B.T @ F_gt @ K_A
        U, _, Vt = np.linalg.svd(E_gt)
        E_gt = U @ np.diag([1.0, 1.0, 0.0]) @ Vt  # enforce rank-2

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = U[:, 2]
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        if np.linalg.det(R1) < 0:
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2

        # Fixed synthetic 3D points in front of camera A (seeded for reproducibility)
        rng = np.random.default_rng(0)
        pts3d = rng.standard_normal((20, 3))
        pts3d[:, 2] = np.abs(pts3d[:, 2]) + 2.0  # guaranteed z > 0 in camera A

        best_R, best_t, best_votes = R1, t[:, None], -1
        for R_cand in (R1, R2):
            for t_sign in (1.0, -1.0):
                t_cand = t_sign * t
                depths_B = (R_cand @ pts3d.T + t_cand[:, None])[2]
                votes = int((depths_B > 0).sum())
                if votes > best_votes:
                    best_votes = votes
                    best_R, best_t = R_cand, t_cand[:, None]

        return best_R, best_t

    # ------------------------------------------------------------------
    # public API (matches OpticalmapHomogBenchmark interface)
    # ------------------------------------------------------------------

    def benchmark(self, model, model_name=None, vis=False, thresh_score=0.05):
        pose_errors = []

        for count, idx in enumerate(tqdm(self.test_idx, desc=f"bench[{self.split}]")):
            if vis and count > 20:
                break

            im_A_path = osp.join(self.data_root, f"pair{idx}_1.jpg")
            im_B_path = osp.join(self.data_root, f"pair{idx}_2.jpg")
            F_gt = np.loadtxt(osp.join(self.data_root, f"gt_{idx}.txt"))
            K_A  = np.loadtxt(osp.join(self.data_root, f"ka_{idx}.txt"))
            K_B  = np.loadtxt(osp.join(self.data_root, f"kb_{idx}.txt"))

            with Image.open(im_A_path) as img:
                w1, h1 = img.size
            with Image.open(im_B_path) as img:
                w2, h2 = img.size

            dense_matches, dense_certainty = model.match(im_A_path, im_B_path)
            sparse_matches, _ = model.sample(
                dense_matches, dense_certainty, 5000, thresh_score=thresh_score
            )
            sm = sparse_matches.cpu().numpy()

            if sm.shape[0] < 8:
                pose_errors.append(180.0)
                continue

            pos_a = self._to_pixel(sm[:, :2], w1, h1)
            pos_b = self._to_pixel(sm[:, 2:], w2, h2)

            R_pred, t_pred, pts_a_n, pts_b_n = self._estimate_pose(pos_a, pos_b, K_A, K_B)
            if R_pred is None:
                pose_errors.append(180.0)
                continue

            R_gt, t_gt = self._gt_pose_from_F(F_gt, K_A, K_B)

            err_R = float(angle_error_mat(R_pred, R_gt))
            err_t = float(angle_error_vec(t_pred.ravel(), t_gt.ravel()))
            err_t = min(err_t, 180.0 - err_t)
            pose_errors.append(max(err_R, err_t))

        thresholds = [5, 10, 20]
        auc = pose_auc(np.array(pose_errors), thresholds)
        return {
            "auc_5":           auc[0],
            "auc_10":          auc[1],
            "auc_20":          auc[2],
            "mean_pose_error": float(np.mean(pose_errors)),
        }

    def benchmark_fundamental_zeroshot(self, model, thresh_score=0.05, grid_size=50):
        """Zero-shot F estimation: no camera intrinsics used.

        Estimates F in pixel space via USAC_MAGSAC from model matches, then
        evaluates against F_gt using epipolar-line angular error on a uniform
        grid of points in image A.  Reports AUC at [1, 2, 5, 10] degree thresholds.
        """
        ele_errors = []  # per-pair mean epipolar-line angular error (degrees)

        for idx in tqdm(self.test_idx, desc=f"F-zeroshot[{self.split}]"):
            im_A_path = osp.join(self.data_root, f"pair{idx}_1.jpg")
            im_B_path = osp.join(self.data_root, f"pair{idx}_2.jpg")
            F_gt = np.loadtxt(osp.join(self.data_root, f"gt_{idx}.txt"))

            with Image.open(im_A_path) as img:
                w1, h1 = img.size
            with Image.open(im_B_path) as img:
                w2, h2 = img.size

            dense_matches, dense_certainty = model.match(im_A_path, im_B_path)
            sparse_matches, _ = model.sample(
                dense_matches, dense_certainty, 5000, thresh_score=thresh_score
            )
            sm = sparse_matches.cpu().numpy()
            pos_a = self._to_pixel(sm[:, :2], w1, h1)
            pos_b = self._to_pixel(sm[:, 2:], w2, h2)

            if len(pos_a) < 8:
                ele_errors.append(180.0)
                continue

            try:
                F_pred, _ = cv2.findFundamentalMat(
                    pos_a.astype(np.float64),
                    pos_b.astype(np.float64),
                    method=cv2.USAC_MAGSAC,
                    confidence=0.99999,
                    ransacReprojThreshold=1.0,
                )
            except Exception:
                F_pred = None

            if F_pred is None or F_pred.shape != (3, 3):
                ele_errors.append(180.0)
                continue

            # Uniform grid in image A
            xs = np.linspace(0, w1 - 1, grid_size)
            ys = np.linspace(0, h1 - 1, grid_size)
            xx, yy = np.meshgrid(xs, ys)
            pts = np.stack([xx.ravel(), yy.ravel(), np.ones(grid_size * grid_size)], axis=-1)  # (N, 3)

            l_gt   = (F_gt   @ pts.T).T  # (N, 3)
            l_pred = (F_pred @ pts.T).T

            # Normalise to unit length in the line direction (first two components)
            norm_gt   = np.linalg.norm(l_gt[:, :2],   axis=-1, keepdims=True) + 1e-12
            norm_pred = np.linalg.norm(l_pred[:, :2], axis=-1, keepdims=True) + 1e-12
            l_gt_n   = l_gt   / norm_gt
            l_pred_n = l_pred / norm_pred

            cos_sim = np.clip(np.abs(np.sum(l_gt_n * l_pred_n, axis=-1)), 0.0, 1.0)
            angles = np.degrees(np.arccos(cos_sim))
            ele_errors.append(float(np.mean(angles)))

        ele_errors = np.array(ele_errors)
        thresholds = [1, 2, 5, 10]
        auc = pose_auc(ele_errors, thresholds)
        return {
            "auc_1":    auc[0],
            "auc_2":    auc[1],
            "auc_5":    auc[2],
            "auc_10":   auc[3],
            "mean_ele": float(np.mean(ele_errors[ele_errors < 180.0])),
        }

    def benchmark_uncertainty(self, model, thresh_score=0.05):
        out = []
        for idx in tqdm(self.test_idx, desc="uncertainty"):
            im_A_path = osp.join(self.data_root, f"pair{idx}_1.jpg")
            im_B_path = osp.join(self.data_root, f"pair{idx}_2.jpg")
            _, dense_certainty = model.match(im_A_path, im_B_path)
            u = 1.0 - float(dense_certainty.mean().item())
            out.append([int(idx), u])
        return out
