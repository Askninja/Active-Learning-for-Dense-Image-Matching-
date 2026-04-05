"""Shared utility functions for active learning sampling strategies."""

import math
import numpy as np
import torch


def log_strategy_action(message: str):
    print(f"[STRATEGY] {message}", flush=True)


# ---------------------------------------------------------------------------
# Homography utilities
# ---------------------------------------------------------------------------

def compute_mean_homography(homographies: np.ndarray) -> np.ndarray:
    """Average the K RANSAC homographies for each pair.

    Args:
        homographies: (N, K, 3, 3) — K homographies per pair.

    Returns:
        (N, 3, 3) — mean homography per pair.
    """
    return homographies.mean(axis=1)


def compute_eigenvalue_features(mean_homographies: np.ndarray) -> np.ndarray:
    """Represent each mean homography by the magnitudes and phases of its eigenvalues.

    The three eigenvalues are sorted by descending magnitude for permutation stability.
    Each eigenvalue contributes |λ| (magnitude) and arg(λ) (phase), giving 6 features.

    Args:
        mean_homographies: (N, 3, 3).

    Returns:
        (N, 6) — [|λ1|, |λ2|, |λ3|, arg(λ1), arg(λ2), arg(λ3)].
    """
    N = mean_homographies.shape[0]
    features = np.zeros((N, 6), dtype=np.float64)
    for i in range(N):
        try:
            eigvals = np.linalg.eigvals(mean_homographies[i])
            order = np.argsort(np.abs(eigvals))[::-1]
            eigvals = eigvals[order]
            features[i, :3] = np.abs(eigvals)
            features[i, 3:] = np.angle(eigvals)
        except np.linalg.LinAlgError:
            features[i] = 0.0
    return features


def compute_displacement_features(mean_homographies: np.ndarray, image_size: int = 560) -> np.ndarray:
    """Represent each mean homography by how it displaces the four image corners.

    Displacements are normalized by image_size so the vector is scale-invariant.

    Args:
        mean_homographies: (N, 3, 3).
        image_size: side length of the square image in pixels (default 560).

    Returns:
        (N, 8) — [δ1_x, δ1_y, δ2_x, δ2_y, δ3_x, δ3_y, δ4_x, δ4_y].
    """
    s = float(image_size)
    corners = np.array([[0, 0], [s, 0], [0, s], [s, s]], dtype=np.float64)
    N = mean_homographies.shape[0]
    features = np.zeros((N, 8), dtype=np.float64)
    for i in range(N):
        H = mean_homographies[i]
        for j, corner in enumerate(corners):
            pt_h = np.array([corner[0], corner[1], 1.0])
            proj_h = H @ pt_h
            denom = proj_h[2] if abs(proj_h[2]) >= 1e-10 else 1e-10
            proj = proj_h[:2] / denom
            delta = np.clip((proj - corner) / s, -10.0, 10.0)
            features[i, 2 * j]     = delta[0]
            features[i, 2 * j + 1] = delta[1]
    return features


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance across the pool.

    Dimensions with near-zero variance are left unchanged to avoid numerical blow-up.

    Args:
        features: (N, D).

    Returns:
        (N, D) — normalized features.
    """
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (features - mean) / std


def k_center_greedy(
    embeddings: np.ndarray,
    b: int,
    initial_idx: np.ndarray = None,
) -> np.ndarray:
    """K-center greedy coreset selection.

    Iteratively selects the point maximally distant from the current selected set.
    Seeds from `initial_idx` if provided, otherwise from the point with highest L2 norm.

    Args:
        embeddings:  (N, D) feature matrix.
        b:           number of samples to select.
        initial_idx: optional (M,) array of pre-selected indices to seed from.

    Returns:
        (b,) integer indices into the input pool.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    N = embeddings.shape[0]
    if N == 0 or b <= 0:
        return np.empty(0, dtype=int)
    b = min(b, N)

    selected = []
    if initial_idx is not None:
        initial_idx = np.asarray(initial_idx, dtype=int)
        selected = [int(i) for i in initial_idx.tolist() if 0 <= int(i) < N]
    if not selected:
        selected = [int(np.argmax(np.sum(embeddings * embeddings, axis=1)))]

    selected_mask = np.zeros(N, dtype=bool)
    min_dist = np.full(N, np.inf, dtype=np.float32)
    for idx in selected:
        selected_mask[idx] = True
        dist = np.linalg.norm(embeddings - embeddings[idx], axis=1).astype(np.float32)
        np.minimum(min_dist, dist, out=min_dist)
    min_dist[selected_mask] = 0.0

    while len(selected) < b:
        next_idx = int(np.argmax(min_dist))
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        dist = np.linalg.norm(embeddings - embeddings[next_idx], axis=1).astype(np.float32)
        np.minimum(min_dist, dist, out=min_dist)
        min_dist[selected_mask] = 0.0

    return np.asarray(selected, dtype=int)


def normalize_weights(values: np.ndarray) -> np.ndarray:
    """Map scores to weights in [1, 2] for uncertainty-weighted selection.

    Args:
        values: (N,) scalar scores.

    Returns:
        (N,) weights in [1, 2].
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    return 1.0 + (values - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Entropy utilities
# ---------------------------------------------------------------------------

def _ensure_entropy_logits_layout(gm_cls: np.ndarray) -> np.ndarray:
    gm_cls = np.asarray(gm_cls, dtype=np.float64)
    if gm_cls.ndim == 3:
        if gm_cls.shape[0] > gm_cls.shape[-1]:
            gm_cls = np.moveaxis(gm_cls, 0, -1)
        gm_cls = gm_cls[None, ...]
    elif gm_cls.ndim == 4:
        if gm_cls.shape[1] > gm_cls.shape[-1]:
            gm_cls = np.moveaxis(gm_cls, 1, -1)
    else:
        raise ValueError(f"gm_cls must have 3 or 4 dims, got shape {gm_cls.shape}")
    return gm_cls


def entropy_from_gm_cls(gm_cls: np.ndarray, temperature: float = 0.5, eps: float = 1e-12) -> np.ndarray:
    logits = _ensure_entropy_logits_layout(gm_cls)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / float(temperature)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), eps)
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    k = probs.shape[-1]
    if k <= 1:
        return np.zeros(entropy.shape, dtype=np.float64)
    return entropy / np.log(k)


def mean_entropy_score(
    gm_cls: np.ndarray,
    matchability: np.ndarray = None,
    temperature: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    entropy = entropy_from_gm_cls(gm_cls, temperature=temperature, eps=eps)
    if matchability is not None:
        matchability = np.asarray(matchability)
        if matchability.ndim == 2:
            matchability = matchability[None, ...]
        if matchability.shape != entropy.shape:
            raise ValueError(
                f"matchability shape {matchability.shape} must match entropy shape {entropy.shape}"
            )
        valid = matchability > 0
    else:
        valid = np.ones_like(entropy, dtype=bool)
    scores = np.zeros(entropy.shape[0], dtype=np.float64)
    for idx in range(entropy.shape[0]):
        values = entropy[idx][valid[idx]]
        scores[idx] = float(values.mean()) if values.size > 0 else 0.0
    return scores if scores.shape[0] > 1 else scores[0]


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _ensure_flow_layout(flow: torch.Tensor) -> torch.Tensor:
    if flow.ndim != 4:
        raise ValueError(f"flow must have 4 dims, got shape {tuple(flow.shape)}")
    if flow.shape[1] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(0, 3, 1, 2)
    raise ValueError(f"flow must have 2 channels, got shape {tuple(flow.shape)}")


def _ensure_confidence_layout(confidence: torch.Tensor) -> torch.Tensor:
    if confidence.ndim == 4 and confidence.shape[1] == 1:
        return confidence[:, 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        return confidence[..., 0]
    if confidence.ndim == 3:
        return confidence
    raise ValueError(f"confidence must have shape (B,H,W) or (B,1,H,W), got {tuple(confidence.shape)}")


def compute_geometry_descriptor(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    bins: int = 16,
    conf_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    flow = _ensure_flow_layout(flow).float()
    confidence = _ensure_confidence_layout(confidence).float()
    if flow.shape[0] != confidence.shape[0] or flow.shape[-2:] != confidence.shape[-2:]:
        raise ValueError(
            f"flow shape {tuple(flow.shape)} and confidence shape {tuple(confidence.shape)} are incompatible"
        )
    batch_size, _, height, width = flow.shape
    device = flow.device
    diag = math.sqrt(height ** 2 + width ** 2)

    flow_x, flow_y = flow[:, 0], flow[:, 1]
    mag = torch.sqrt(flow_x.square() + flow_y.square()).div_(diag + eps).clamp_(0.0, 1.0)
    theta = torch.atan2(flow_y, flow_x)

    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    weights = confidence * (confidence > conf_threshold)

    mag_bins = torch.clamp((mag * bins).long(), 0, bins - 1)
    dir_bins = torch.clamp(((theta + math.pi) / (2.0 * math.pi + eps) * bins).long(), 0, bins - 1)

    flat_weights = weights.reshape(batch_size, -1)
    mag_hist = torch.zeros(batch_size, bins, device=device, dtype=torch.float32)
    dir_hist = torch.zeros(batch_size, bins, device=device, dtype=torch.float32)
    mag_hist.scatter_add_(1, mag_bins.reshape(batch_size, -1), flat_weights)
    dir_hist.scatter_add_(1, dir_bins.reshape(batch_size, -1), flat_weights)
    mag_hist.div_(mag_hist.sum(dim=1, keepdim=True) + eps)
    dir_hist.div_(dir_hist.sum(dim=1, keepdim=True) + eps)
    coverage = (confidence > conf_threshold).float().mean(dim=(1, 2)).unsqueeze(1)

    return torch.cat((mag_hist, dir_hist, coverage), dim=1)


def geometry_novelty_scores(
    query_descriptors: torch.Tensor,
    labeled_descriptors: torch.Tensor,
    eps: float = 1e-8,
    chunk_size: int = 2048,
) -> torch.Tensor:
    query_descriptors = query_descriptors.float()
    labeled_descriptors = labeled_descriptors.float()
    if query_descriptors.ndim != 2 or labeled_descriptors.ndim != 2:
        raise ValueError("descriptors must be 2D")
    if query_descriptors.shape[1] != labeled_descriptors.shape[1]:
        raise ValueError("descriptor dimensions must match")
    if labeled_descriptors.shape[0] == 0:
        return torch.linalg.norm(query_descriptors, dim=1)
    scores = []
    for start in range(0, query_descriptors.shape[0], max(1, int(chunk_size))):
        chunk = query_descriptors[start:start + chunk_size]
        scores.append(torch.cdist(chunk, labeled_descriptors, p=2).min(dim=1).values)
    return torch.cat(scores, dim=0) if scores else torch.empty(0, device=query_descriptors.device)
