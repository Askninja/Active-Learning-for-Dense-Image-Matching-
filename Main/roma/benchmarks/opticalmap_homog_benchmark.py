from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from roma.utils import pose_auc
from roma.utils import *
from roma.utils.kde import kde
from scipy.stats import pearsonr
import os.path as osp
import pandas as pd
from matplotlib.patches import Polygon
from roma.utils.utils import cls_to_flow_refine
import torch.nn.functional as F
from torchvision import transforms

class OpticalmapHomogBenchmark:
    def __init__(self, data_root, split) -> None:
        self.data_root = data_root
        self.test_idx = np.load(osp.join(self.data_root, f'{split}.npy'))
        self.split = split
        print(f'{split} size is {self.test_idx.size}')

    def convert_coordinates(self, im_A_coords, im_A_to_im_B, wq, hq, wsup, hsup):
        offset = 0.5
        im_A_coords = (
            np.stack(
                (
                    wq * (im_A_coords[..., 0] + 1) / 2,
                    hq * (im_A_coords[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        im_A_to_im_B = (
            np.stack(
                (
                    wsup * (im_A_to_im_B[..., 0] + 1) / 2,
                    hsup * (im_A_to_im_B[..., 1] + 1) / 2,
                ),
                axis=-1,
            )
            - offset
        )
        return im_A_coords, im_A_to_im_B

    def load_im(self, im_B):
        im = cv2.imread(im_B)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        return im

    @staticmethod
    def _hs_from_model(model, im_A_path, im_B_path, iters=50, n_sample=5000, thresh_score=0.05):
        with Image.open(im_A_path) as imA_pil:
            w1, h1 = imA_pil.size
        with Image.open(im_B_path) as imB_pil:
            w2, h2 = imB_pil.size
        dense_matches, dense_certainty = model.match(im_A_path, im_B_path)
        sparse_matches, _ = model.sample(dense_matches, dense_certainty, n_sample, thresh_score=thresh_score)
        sm = sparse_matches.detach().cpu().numpy()
        if sm.shape[0] < 8:
            return 0.0
        def to_px(coords, w, h):
            offset = 0.5
            x = w * (coords[..., 0] + 1) / 2.0 - offset
            y = h * (coords[..., 1] + 1) / 2.0 - offset
            return np.stack([x, y], axis=-1)
        A_px = to_px(sm[:, :2], w1, h1)
        B_px = to_px(sm[:, 2:], w2, h2)
        Hs = []
        g = np.random.default_rng(1234)
        subset = min(2000, A_px.shape[0])
        for _ in range(iters):
            if A_px.shape[0] > subset:
                sel = g.choice(A_px.shape[0], size=subset, replace=False)
                pA = A_px[sel]
                pB = B_px[sel]
            else:
                pA = A_px
                pB = B_px
            H, _ = cv2.findHomography(pA, pB, method=cv2.RANSAC, ransacReprojThreshold=3 * min(w2, h2) / 480, confidence=0.999)
            if H is not None and abs(H[2, 2]) > 1e-12:
                Hs.append(H / (H[2, 2] + 1e-12))
        if len(Hs) < 2:
            return 0.0
        Hs = np.stack(Hs, axis=0)
        corners = np.float32([[0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]]).reshape(-1, 1, 2)
        warped = []
        for H in Hs:
            wc = cv2.perspectiveTransform(corners, H).reshape(4, 2)
            warped.append(wc)
        warped = np.stack(warped, axis=0)
        std_xy = warped.std(axis=0)
        score = float(std_xy.mean())
        return score

    def benchmark(self, model, model_name=None, vis=False, thresh_score=0.05):
        homog_dists = []
        all_epe = []
        for count, idx in enumerate(self.test_idx):
            if vis and count > 20:
                break
            optical_name = f'pair{idx}_1'
            map_name = f'pair{idx}_2'
            optical_path = os.path.join(self.data_root, optical_name + '.jpg')
            map_path = os.path.join(self.data_root, map_name + '.jpg')
            homo_path = os.path.join(self.data_root, f'gt_{idx}.txt')
            homo = np.loadtxt(homo_path)
            if homo.shape[0] == 2:
                homo = np.vstack([homo, np.array([0, 0, 1])])
            H_gt = torch.tensor(homo, dtype=torch.float)
            im_A_path = optical_path
            im_A = self.load_im(im_A_path)
            w1, h1 = im_A.size
            im_B_path = map_path
            im_B = self.load_im(im_B_path)
            w2, h2 = im_B.size
            dense_matches, dense_certainty = model.match(im_A_path, im_B_path)
            sparse_matches, sparse_certainty = model.sample(
                dense_matches, dense_certainty, 5000, thresh_score=thresh_score
            )
            sparse_matches_np = sparse_matches.cpu().numpy()
            pos_a, pos_b = self.convert_coordinates(
                sparse_matches_np[:, :2], sparse_matches_np[:, 2:], w1, h1, w2, h2
            )
            try:
                H_pred, inliers = cv2.findHomography(
                    pos_a,
                    pos_b,
                    method=cv2.RANSAC,
                    confidence=0.99999,
                    ransacReprojThreshold=3 * min(w2, h2) / 480,
                )
            except Exception:
                H_pred = None
            if H_pred is None:
                H_pred = np.zeros((3, 3))
                H_pred[2, 2] = 1.0
            corners = np.array(
                [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]]
            )
            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = (
                real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            )
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            mean_dist = np.mean(
                np.linalg.norm(real_warped_corners - warped_corners, axis=1)
            ) / (min(w2, h2) / 480.0)
            homog_dists.append(mean_dist)
            dense_matches_np = dense_matches.reshape(-1, 4).cpu().numpy()
            dense_a, dense_b = self.convert_coordinates(
                dense_matches_np[:, :2], dense_matches_np[:, 2:], w1, h1, w2, h2
            )
            w_dense_a = np.dot(
                np.concatenate([dense_a, np.ones_like(dense_a[:, 0:1])], axis=1),
                np.transpose(H_gt),
            )
            w_dense_a = w_dense_a / w_dense_a[:, 2:]
            epe = np.mean(np.linalg.norm(w_dense_a[:, :2] - dense_b, axis=1)) / (
                min(w2, h2) / 480.0
            )
            all_epe.append(epe)
            uncert = 'default'
            certainty = dense_certainty
            H, W2, _ = dense_matches.shape
            W = W2 // 2 if model.symmetric else W2
            if vis:
                save_dir = os.path.join('vis', model_name + '_' + self.split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                vis_warp = model.visualize_warp(
                    dense_matches, certainty, im_A_path=im_A_path, im_B_path=im_B_path
                )
                tensor_to_pil(vis_warp, unnormalize=False).save(
                    f"{save_dir}/{count}_warp_error_{mean_dist:.2f}.jpg"
                )
                vis_certainty = certainty.cpu().detach().numpy().squeeze()
                fig = plt.figure()
                plt.imshow(vis_certainty, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_certainty_{uncert}.jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                kpts1, kpts2 = model.to_pixel_coordinates(sparse_matches, H, W, H, W)
                kpts1, kpts2 = (
                    kpts1.detach().cpu().numpy().astype(np.int32),
                    kpts2.detach().cpu().numpy().astype(np.int32),
                )
                vis_certainty = np.uint8(certainty.cpu().detach().numpy().squeeze() * 255)
                vis_certainty = vis_certainty[:, :, None]
                vis_certainty = np.tile(vis_certainty, (1, 1, 3))
                covisible_mask = (
                    (w_dense_a[:, 0] > 0)
                    * (w_dense_a[:, 0] < w2 - 1)
                    * (w_dense_a[:, 1] > 0)
                    * (w_dense_a[:, 1] < h2 - 1)
                )
                covisible_mask = covisible_mask.astype(np.float32).reshape(H, W)
                fig = plt.figure()
                plt.imshow(covisible_mask, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_covisible_mask.jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                fig = plt.figure()
                plt.imshow(255 * np.ones_like(vis_certainty, dtype=np.uint8))
                conf = sparse_certainty.detach().cpu().numpy()
                colors = cm.jet(conf, alpha=1)
                plt.scatter(kpts1[:, 0], kpts1[:, 1], marker="x", c=colors, s=0.1)
                plt.scatter(W + kpts2[:, 0], kpts2[:, 1], marker="x", c=colors, s=0.1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_sample_points_error_{mean_dist:.2f}.jpg",
                    format="jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                im_A_name = os.path.splitext(os.path.basename(im_A_path))[0]
                im_B_name = os.path.splitext(os.path.basename(im_B_path))[0]
                cat_imgAB = Image.fromarray(
                    np.hstack((np.array(im_A.resize((W, H))), np.array(im_B.resize((W, H)))))
                )
                cat_imgAB.save(f"{save_dir}/{count}_imAB.jpg")
                im_A_warp_gt = Image.fromarray(
                    cv2.warpPerspective(np.array(im_A), np.array(H_gt), im_B.size)
                )
                im_A_warp_gt.save(f"{save_dir}/{count}_imA_warp_gt.jpg")
                im_A_warp_pred = Image.fromarray(
                    cv2.warpPerspective(np.array(im_A), np.array(H_pred), im_B.size)
                )
                im_A_warp_pred.save(f"{save_dir}/{count}_imA_warp_pred_error_{mean_dist:.2f}.jpg")
                p_gt = Polygon(real_warped_corners, facecolor='none', edgecolor='blue', label='gt')
                p_pred = Polygon(warped_corners, facecolor='none', edgecolor='red', label='pred')
                fig, ax = plt.subplots(1, 1)
                ax.add_patch(p_gt)
                ax.add_patch(p_pred)
                plt.ylim(-100, 1500)
                plt.xlim(-100, 1500)
                plt.show()
                plt.legend()
                plt.savefig(
                    f"{save_dir}/{count}_corner_error_{mean_dist:.2f}.jpg",
                    format="jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                im_A.save(f"{save_dir}/{count}_imA.jpg")
                im_B.save(f"{save_dir}/{count}_imB.jpg")
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        auc = pose_auc(np.array(homog_dists), thresholds)
        return {
            "auc_3": auc[2],
            "auc_5": auc[4],
            "auc_10": auc[9],
            "epe": np.mean(all_epe),
        }

    def convert_to_matches(self, im_A_to_im_B, certainty, scl, hs, ws, symmetric):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if scl != 1:
            im_A_to_im_B = F.interpolate(
                im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
            )
            certainty = F.interpolate(
                certainty, size=(hs, ws), align_corners=False, mode="bilinear"
            )
        im_A_to_im_B = im_A_to_im_B.permute(0, 2, 3, 1)
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
        im_A_coords = im_A_coords[None].expand(1, 2, hs, ws)
        certainty = certainty.sigmoid()
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)
        if (im_A_to_im_B.abs() > 1).any() and True:
            wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
            certainty[wrong[:, None]] = 0
        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        if symmetric:
            A_to_B, B_to_A = im_A_to_im_B.chunk(2)
            q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
            im_B_coords = im_A_coords
            s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
            warp = torch.cat((q_warp, s_warp), dim=2)
            certainty = torch.cat(certainty.chunk(2), dim=3)
        else:
            warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        return warp[0], certainty[0, 0]

    def get_error(self, model, dense_matches, dense_certainty, n_sample, thresh_score, w1, h1, w2, h2, H_gt):
        sparse_matches, sparse_certainty = model.sample(
            dense_matches, dense_certainty, 5000, thresh_score=thresh_score
        )
        sparse_matches_np = sparse_matches.cpu().numpy()
        pos_a, pos_b = self.convert_coordinates(
            sparse_matches_np[:, :2], sparse_matches_np[:, 2:], w1, h1, w2, h2
        )
        try:
            H_pred, inliers = cv2.findHomography(
                pos_a,
                pos_b,
                method=cv2.RANSAC,
                confidence=0.99999,
                ransacReprojThreshold=3 * min(w2, h2) / 480,
            )
        except Exception:
            H_pred = None
        if H_pred is None:
            H_pred = np.zeros((3, 3))
            H_pred[2, 2] = 1.0
        corners = np.array(
            [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]]
        )
        real_warped_corners = np.dot(corners, np.transpose(H_gt))
        real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
        warped_corners = np.dot(corners, np.transpose(H_pred))
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
        mean_dist = np.mean(
            np.linalg.norm(real_warped_corners - warped_corners, axis=1)
        ) / (min(w2, h2) / 480.0)
        return mean_dist

    def vis_corresps(self, model, model_name=None, thresh_score=0.05):
        save_dir = os.path.join('vis', 'corresps_opticalmap_' + model_name + '_' + self.split)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.upsample_preds = False
        model.symmetric = False
        for count, idx in enumerate(self.test_idx):
            if count > 10:
                break
            optical_name = f'pair{idx}_1'
            map_name = f'pair{idx}_2'
            optical_path = os.path.join(self.data_root, optical_name + '.jpg')
            map_path = os.path.join(self.data_root, map_name + '.jpg')
            homo_path = os.path.join(self.data_root, f'gt_{idx}.txt')
            homo = np.loadtxt(homo_path)
            H_gt = torch.tensor(homo, dtype=torch.float)
            im_A_path = optical_path
            im_A = self.load_im(im_A_path)
            w1, h1 = im_A.size
            im_B_path = map_path
            im_B = self.load_im(im_B_path)
            w2, h2 = im_B.size
            corresps = model.get_corresps(im_A_path, im_B_path)
            scales = [1, 2, 4, 8, 16]
            ws = model.w_resized
            hs = model.h_resized
            for scl in scales:
                im_A_to_im_B = corresps[scl]["flow"]
                certainty = corresps[scl]["certainty"]
                dense_matches, dense_certainty = self.convert_to_matches(
                    im_A_to_im_B, certainty, scl, hs, ws, model.symmetric
                )
                error = self.get_error(
                    model, dense_matches, dense_certainty, 5000, thresh_score, w1, h1, w2, h2, H_gt
                )
                vis_warp = model.visualize_warp(
                    dense_matches, dense_certainty, im_A_path=im_A_path, im_B_path=im_B_path, symmetric=model.symmetric
                )
                tensor_to_pil(vis_warp, unnormalize=False).save(
                    f"{save_dir}/{count}_warp_scl_{scl}_error_{error:.2f}.jpg"
                )
                if scl == 16:
                    im_A_to_im_B = cls_to_flow_refine(corresps[scl]["gm_cls"]).permute(0, 3, 1, 2)
                    certainty = corresps[scl]["gm_certainty"]
                    dense_matches, dense_certainty = self.convert_to_matches(
                        im_A_to_im_B, certainty, scl, hs, ws, model.symmetric
                    )
                    error = self.get_error(
                        model, dense_matches, dense_certainty, 5000, thresh_score, w1, h1, w2, h2, H_gt
                    )
                    vis_warp = model.visualize_warp(
                        dense_matches, dense_certainty, im_A_path=im_A_path, im_B_path=im_B_path, symmetric=model.symmetric
                    )
                    tensor_to_pil(vis_warp, unnormalize=False).save(
                        f"{save_dir}/{count}_warp_scl_{scl}gm_error_{error:.2f}.jpg"
                    )

    def benchmark_uncertainty(self, model, model_name=None, vis=False, thresh_score=0.05):
        pairs = []
        homog_dists = []
        all_epe = []
        for count, idx in enumerate(self.test_idx):
            if vis and count > 20:
                break
            optical_name = f'pair{idx}_1'
            map_name = f'pair{idx}_2'
            optical_path = os.path.join(self.data_root, optical_name + '.jpg')
            map_path = os.path.join(self.data_root, map_name + '.jpg')
            homo_path = os.path.join(self.data_root, f'gt_{idx}.txt')
            homo = np.loadtxt(homo_path)
            if homo.shape[0] == 2:
                homo = np.vstack([homo, np.array([0, 0, 1])])
            H_gt = torch.tensor(homo, dtype=torch.float)
            im_A_path = optical_path
            im_A = self.load_im(im_A_path)
            w1, h1 = im_A.size
            im_B_path = map_path
            im_B = self.load_im(im_B_path)
            w2, h2 = im_B.size
            dense_matches, dense_certainty = model.match(im_A_path, im_B_path)
            sparse_matches, sparse_certainty = model.sample(
                dense_matches, dense_certainty, 5000, thresh_score=thresh_score
            )
            sparse_matches_np = sparse_matches.cpu().numpy()
            pos_a, pos_b = self.convert_coordinates(
                sparse_matches_np[:, :2], sparse_matches_np[:, 2:], w1, h1, w2, h2
            )
            try:
                H_pred, inliers = cv2.findHomography(
                    pos_a,
                    pos_b,
                    method=cv2.RANSAC,
                    confidence=0.99999,
                    ransacReprojThreshold=3 * min(w2, h2) / 480,
                )
            except Exception:
                H_pred = None
            if H_pred is None:
                H_pred = np.zeros((3, 3))
                H_pred[2, 2] = 1.0
            corners = np.array(
                [[0, 0, 1], [0, h1 - 1, 1], [w1 - 1, h1 - 1, 1], [w1 - 1, 0, 1]]
            )
            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            mean_dist = np.mean(
                np.linalg.norm(real_warped_corners - warped_corners, axis=1)
            ) / (min(w2, h2) / 480.0)
            homog_dists.append(mean_dist)
            dense_matches_np = dense_matches.reshape(-1, 4).cpu().numpy()
            dense_a, dense_b = self.convert_coordinates(
                dense_matches_np[:, :2], dense_matches_np[:, 2:], w1, h1, w2, h2
            )
            w_dense_a = np.dot(
                np.concatenate([dense_a, np.ones_like(dense_a[:, 0:1])], axis=1),
                np.transpose(H_gt),
            )
            w_dense_a = w_dense_a / w_dense_a[:, 2:]
            epe = np.mean(np.linalg.norm(w_dense_a[:, :2] - dense_b, axis=1)) / (
                min(w2, h2) / 480.0
            )
            all_epe.append(epe)
            certainty = dense_certainty
            Ht, W2, _ = dense_matches.shape
            W = W2 // 2 if getattr(model, "symmetric", False) else W2
            u = 1.0 - float(dense_certainty.mean().item())
            pairs.append([int(idx), float(u)])
            if vis:
                save_dir = os.path.join('vis', (model_name or 'model') + '_' + self.split)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                vis_warp = model.visualize_warp(
                    dense_matches, certainty, im_A_path=im_A_path, im_B_path=im_B_path
                )
                tensor_to_pil(vis_warp, unnormalize=False).save(
                    f"{save_dir}/{count}_warp_error_{mean_dist:.2f}.jpg"
                )
                vis_certainty = certainty.cpu().detach().numpy().squeeze()
                fig = plt.figure()
                plt.imshow(vis_certainty, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_certainty.jpg", dpi=300, bbox_inches="tight", pad_inches=0
                )
                plt.close()
                kpts1, kpts2 = model.to_pixel_coordinates(sparse_matches, Ht, W, Ht, W)
                kpts1 = kpts1.detach().cpu().numpy().astype(np.int32)
                kpts2 = kpts2.detach().cpu().numpy().astype(np.int32)
                vis_c = np.uint8(certainty.cpu().detach().numpy().squeeze() * 255)
                vis_c = np.tile(vis_c[:, :, None], (1, 1, 3))
                covisible_mask = (
                    (w_dense_a[:, 0] > 0)
                    * (w_dense_a[:, 0] < w2 - 1)
                    * (w_dense_a[:, 1] > 0)
                    * (w_dense_a[:, 1] < h2 - 1)
                ).astype(np.float32).reshape(Ht, W)
                fig = plt.figure()
                plt.imshow(covisible_mask, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_covisible_mask.jpg", dpi=300, bbox_inches="tight", pad_inches=0
                )
                plt.close()
                fig = plt.figure()
                plt.imshow(255 * np.ones_like(vis_c, dtype=np.uint8))
                conf = (
                    sparse_certainty.detach().cpu().numpy()
                    if sparse_certainty is not None
                    else np.zeros((0,))
                )
                colors = cm.jet(conf, alpha=1) if conf.size > 0 else None
                if conf.size > 0:
                    plt.scatter(kpts1[:, 0], kpts1[:, 1], marker="x", c=colors, s=0.1)
                    plt.scatter(W + kpts2[:, 0], kpts2[:, 1], marker="x", c=colors, s=0.1)
                plt.axis('off')
                plt.savefig(
                    f"{save_dir}/{count}_sample_points_error_{mean_dist:.2f}.jpg",
                    format="jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
                im_A.save(f"{save_dir}/{count}_imA.jpg")
                im_B.save(f"{save_dir}/{count}_imB.jpg")
        return pairs

    def benchmark_hs_model(self, model, iters=50, n_sample=5000, thresh_score=0.05):
        out = []
        for idx in self.test_idx:
            im_A_path = os.path.join(self.data_root, f'pair{idx}_1.jpg')
            im_B_path = os.path.join(self.data_root, f'pair{idx}_2.jpg')
            hs = self._hs_from_model(model, im_A_path, im_B_path, iters=iters, n_sample=n_sample, thresh_score=thresh_score)
            out.append([int(idx), float(hs)])
        return out
