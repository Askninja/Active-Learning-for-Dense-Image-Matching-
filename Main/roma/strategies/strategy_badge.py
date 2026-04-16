"""BADGE strategy for RoMa dense image matching.

This implementation avoids materializing the full last-layer gradient
(`n_cls * hidden_dim` per sample), which is too large for realistic pool
sizes in RoMa. Instead it computes the BADGE gradient in its natural
matrix form and applies a fixed structured random projection directly to
that matrix before running k-means++.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from roma.strategies.strategy_utils import log_strategy_action


def _find_last_linear(model: nn.Module) -> nn.Linear:
    """Return the decoder output projection that produces coarse gm_cls logits."""
    try:
        layer = model.decoder.embedding_decoder.to_out
        if isinstance(layer, nn.Linear):
            return layer
    except AttributeError:
        pass

    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith("embedding_decoder.to_out"):
            candidates.append((name, module))
    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        return candidates[-1][1]

    raise RuntimeError("BADGE: cannot find the decoder output projection layer.")


def get_gm_cls_logits_and_features(
    model: nn.Module,
    batch: dict,
    last_linear: nn.Linear,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and return coarse logits plus decoder features.

    Returns:
        logits: (B, n_cls, H, W) float32 tensor.
        features: (B, H*W, hidden_dim) float32 tensor fed into `last_linear`.
    """
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, inputs, _output):
        captured["features"] = inputs[0]

    handle = last_linear.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            corresps = model(batch)
    finally:
        handle.remove()

    gm_scales = [scale for scale, payload in corresps.items() if payload.get("gm_cls") is not None]
    if not gm_scales:
        raise RuntimeError(
            "BADGE: model forward pass did not produce gm_cls outputs. "
            "Ensure the model is built with is_classifier=True."
        )
    if "features" not in captured:
        raise RuntimeError("BADGE: failed to capture decoder features before the output projection.")

    gm_scale = max(gm_scales)
    logits = corresps[gm_scale]["gm_cls"].float()
    features = captured["features"].float()

    batch_size, _, height, width = logits.shape
    expected_tokens = height * width
    if features.ndim != 3 or features.shape[0] != batch_size or features.shape[1] != expected_tokens:
        raise RuntimeError(
            "BADGE: decoder feature shape does not match gm_cls layout: "
            f"logits={tuple(logits.shape)}, features={tuple(features.shape)}"
        )

    return logits, features


@dataclass
class BadgeProjector:
    """Structured random projection for matrix-form BADGE embeddings."""

    class_proj: np.ndarray
    feature_proj: np.ndarray

    @property
    def output_dim(self) -> int:
        return int(self.class_proj.shape[1] * self.feature_proj.shape[1])


def _projection_shape(n_classes: int, hidden_dim: int, target_dim: int) -> tuple[int, int]:
    class_dim = max(1, min(n_classes, int(math.sqrt(target_dim))))
    feature_dim = max(1, min(hidden_dim, int(math.ceil(target_dim / class_dim))))
    return class_dim, feature_dim


def make_badge_projector(
    n_classes: int,
    hidden_dim: int,
    target_dim: int = 256,
    rng: np.random.Generator | None = None,
) -> BadgeProjector:
    """Create a fixed structured projection shared by all pool samples."""
    if rng is None:
        rng = np.random.default_rng(42)

    class_dim, feature_dim = _projection_shape(n_classes, hidden_dim, target_dim)
    class_proj = rng.standard_normal((n_classes, class_dim), dtype=np.float32) / math.sqrt(class_dim)
    feature_proj = rng.standard_normal((hidden_dim, feature_dim), dtype=np.float32) / math.sqrt(feature_dim)
    return BadgeProjector(class_proj=class_proj, feature_proj=feature_proj)


def compute_badge_embedding(
    logits: torch.Tensor,
    features: torch.Tensor,
    projector: BadgeProjector,
) -> np.ndarray:
    """Compute a projected BADGE embedding for one image pair.

    BADGE uses the gradient of the hallucinated cross-entropy loss with
    respect to the final linear classifier weights. For a linear layer,
    that gradient is a sum of token-wise outer products between the class
    residuals and the decoder features. We keep that matrix form and
    project it directly instead of flattening the full dense gradient.
    """
    batch_size, n_classes, height, width = logits.shape
    if batch_size != 1:
        raise ValueError(f"BADGE expects batch_size=1 during selection, got {batch_size}.")

    token_count = height * width
    logits_tokens = logits.permute(0, 2, 3, 1).reshape(batch_size, token_count, n_classes)
    probs = torch.softmax(logits_tokens, dim=-1)
    pseudo_labels = logits_tokens.detach().argmax(dim=-1, keepdim=True)
    residuals = probs.clone()
    residuals.scatter_add_(-1, pseudo_labels, torch.full_like(pseudo_labels, -1, dtype=probs.dtype))
    residuals = residuals[0] / float(token_count)   # (T, C)
    token_features = features[0]                    # (T, D)

    class_proj = torch.from_numpy(projector.class_proj).to(device=logits.device, dtype=logits.dtype)
    feature_proj = torch.from_numpy(projector.feature_proj).to(device=features.device, dtype=features.dtype)

    residuals_small = residuals @ class_proj
    features_small = token_features @ feature_proj
    projected = residuals_small.transpose(0, 1) @ features_small
    return projected.reshape(-1).detach().cpu().numpy().astype(np.float32)


def kmeans_plus_plus(
    embeddings: np.ndarray,
    k: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """K-means++ seeding using D^2 sampling over the embedding matrix."""
    if rng is None:
        rng = np.random.default_rng(42)

    num_points = len(embeddings)
    if k >= num_points:
        return np.arange(num_points, dtype=int)

    center_indices = [int(rng.integers(0, num_points))]
    min_sq_dists = np.sum((embeddings - embeddings[center_indices[0]]) ** 2, axis=1, dtype=np.float64)
    min_sq_dists[center_indices[0]] = 0.0

    for _ in range(k - 1):
        total = float(min_sq_dists.sum())
        if total < 1e-12:
            remaining = np.setdiff1d(np.arange(num_points, dtype=int), np.asarray(center_indices, dtype=int))
            if remaining.size == 0:
                break
            next_center = int(rng.choice(remaining))
        else:
            next_center = int(rng.choice(num_points, p=min_sq_dists / total))
        center_indices.append(next_center)
        sq_dists = np.sum((embeddings - embeddings[next_center]) ** 2, axis=1, dtype=np.float64)
        np.minimum(min_sq_dists, sq_dists, out=min_sq_dists)
        min_sq_dists[next_center] = 0.0

    return np.asarray(center_indices[:k], dtype=int)


def run(strategy, budget: int, model: nn.Module) -> np.ndarray:
    """Select `budget` unlabeled pairs with BADGE."""
    if model is None:
        raise ValueError("model is required for badge strategy")

    avail = strategy.remaining()
    if avail.size == 0 or budget <= 0:
        return np.empty(0, dtype=int)
    budget = min(int(budget), avail.size)

    last_linear = _find_last_linear(model)
    model.eval()
    device = next(model.parameters()).device

    dataset = strategy._entropy_dataset(avail)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    embeddings: list[np.ndarray] = []
    valid_ids: list[int] = []
    projector: BadgeProjector | None = None
    rng = getattr(strategy, "rng", None)

    for pair_id, batch in zip(avail.tolist(), dataloader):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        try:
            logits, features = get_gm_cls_logits_and_features(model, batch, last_linear)
            if projector is None:
                projector = make_badge_projector(
                    n_classes=int(logits.shape[1]),
                    hidden_dim=int(features.shape[-1]),
                    target_dim=256,
                    rng=rng,
                )
                log_strategy_action(
                    "BADGE: using projected embedding "
                    f"{logits.shape[1]}x{features.shape[-1]} -> {projector.output_dim}."
                )
            emb = compute_badge_embedding(logits, features, projector)
            embeddings.append(emb)
            valid_ids.append(int(pair_id))
        except Exception as exc:
            log_strategy_action(f"BADGE: skipping pair {pair_id} - {exc}")
        finally:
            model.zero_grad(set_to_none=True)

    if not embeddings:
        return np.empty(0, dtype=int)

    emb_matrix = np.asarray(embeddings, dtype=np.float32)
    selected_pos = kmeans_plus_plus(emb_matrix, budget, rng=rng)
    log_strategy_action(
        f"BADGE: scored {len(emb_matrix)} pairs, embedding_dim={emb_matrix.shape[1]}, selected {len(selected_pos)}."
    )
    return np.asarray(valid_ids, dtype=int)[selected_pos]
