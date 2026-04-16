"""Loss Prediction Module for Learning Loss active learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class LossPredictionModule(nn.Module):
    """Predict per-pair RoMa training loss from pooled decoder features.

    The module is trained jointly with RoMa using a pairwise margin ranking loss
    so that higher predicted scores correspond to harder training pairs.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted scalar losses for pooled decoder features."""
        return self.net(x)

    @staticmethod
    def ranking_loss(
        pred_losses: torch.Tensor,
        true_losses: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss from Yoo and Kweon (CVPR 2019)."""
        pred_losses = pred_losses.reshape(-1)
        true_losses = true_losses.reshape(-1)
        B = int(pred_losses.shape[0])
        if B < 2:
            return pred_losses.new_zeros(())

        pred_i = pred_losses.unsqueeze(1).expand(B, B)
        pred_j = pred_losses.unsqueeze(0).expand(B, B)
        true_i = true_losses.unsqueeze(1).expand(B, B)
        true_j = true_losses.unsqueeze(0).expand(B, B)

        sign_ij = torch.sign(true_i - true_j)
        valid = ~torch.eye(B, dtype=torch.bool, device=pred_losses.device)
        valid = valid & (sign_ij != 0)
        if not torch.any(valid):
            return pred_losses.new_zeros(())

        loss_matrix = torch.clamp(-sign_ij * (pred_i - pred_j) + margin, min=0.0)
        return loss_matrix[valid].mean()
