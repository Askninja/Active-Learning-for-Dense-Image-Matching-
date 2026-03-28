import os
import random
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf
from torchvision.transforms import transforms
import os.path as osp
from roma.utils import get_tuple_transform_ops


class OpticalMap(Dataset):
    def __init__(
        self,
        data_root,
        ht=384,
        wt=512,
        use_horizontal_flip_aug=False,
        use_cropping_aug=False,
        min_crop_ratio=0.5,
        use_color_jitter_aug=False,
        use_swap_aug=False,
        use_dual_cropping_aug=False,
        split="train",
    ) -> None:
        self.data_root = data_root
        self.train_idx = np.load(osp.join(data_root, f"{split}.npy"))
        print(f"{split} size is {self.train_idx.size}")

        self.im_transform_ops = get_tuple_transform_ops(resize=(ht, wt), normalize=True)
        self.wt, self.ht = wt, ht
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_cropping_aug = use_cropping_aug
        self.min_crop_ratio = min_crop_ratio
        self.use_color_jitter_aug = use_color_jitter_aug
        self.use_swap_aug = use_swap_aug
        self.use_dual_cropping_aug = use_dual_cropping_aug

    def load_im(self, im_path):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return Image.fromarray(im)

    def __len__(self):
        return self.train_idx.size

    def scale_intrinsic(self, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        return torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=torch.float)

    def horizontal_flip(self, im_A, im_B, H):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        flip_mat = torch.tensor([[-1, 0, self.wt], [0, 1, 0], [0, 0, 1.]], dtype=H.dtype, device=H.device)
        H = torch.linalg.inv(flip_mat) @ H @ flip_mat
        return im_A, im_B, H

    def crop_im(self, im):
        max_ratio, min_ratio = 0.95, self.min_crop_ratio
        ratio = np.random.rand() * (max_ratio - min_ratio) + min_ratio
        w, h = im.size
        crop_w, crop_h = int(w * ratio), int(h * ratio)
        left = np.random.randint(0, w - crop_w + 1)
        upper = np.random.randint(0, h - crop_h + 1)
        im = im.crop((left, upper, left + crop_w, upper + crop_h))
        T = torch.tensor([[1, 0, -left], [0, 1, -upper], [0, 0, 1]], dtype=torch.float)
        return im, T

    def _ensure_homog(self, H: torch.Tensor) -> torch.Tensor:
        """Ensure H is 3×3 homogeneous."""
        if H.shape == (2, 3):
            H = torch.cat([H, torch.tensor([[0., 0., 1.]], dtype=H.dtype)], dim=0)
        return H

    def __getitem__(self, idx):
        pair_id = int(self.train_idx[idx])
        optical_path = osp.join(self.data_root, f"pair{pair_id}_1.jpg")
        map_path = osp.join(self.data_root, f"pair{pair_id}_2.jpg")
        homo_path = osp.join(self.data_root, f"gt_{pair_id}.txt")

        H = torch.tensor(np.loadtxt(homo_path), dtype=torch.float)
        H = self._ensure_homog(H)

        if self.use_swap_aug and np.random.rand() > 0.5:
            imA_path, imB_path = map_path, optical_path
            H = torch.linalg.inv(H)
        else:
            imA_path, imB_path = optical_path, map_path

        im_A = self.load_im(imA_path)
        im_B = self.load_im(imB_path)

        if self.use_cropping_aug:
            im_A, T_A = self.crop_im(im_A)
            H = H @ torch.linalg.inv(T_A)

        if self.use_dual_cropping_aug:
            im_B, T_B = self.crop_im(im_B)
            H = T_B @ H

        K_A = self.scale_intrinsic(im_A.width, im_A.height)
        K_B = self.scale_intrinsic(im_B.width, im_B.height)
        H = K_B @ H @ torch.linalg.inv(K_A)

        if self.use_color_jitter_aug:
            cj = transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)
            im_A, im_B = cj(im_A), cj(im_B)

        im_A, im_B = self.im_transform_ops((im_A, im_B))

        if self.use_horizontal_flip_aug and np.random.rand() > 0.5:
            im_A, im_B, H = self.horizontal_flip(im_A, im_B, H)

        H = self._ensure_homog(H)
        H_inv = torch.linalg.inv(H)

        return {
            "im_A": im_A,
            "im_B": im_B,
            "H": H,
            "H_inv": H_inv,
        }
