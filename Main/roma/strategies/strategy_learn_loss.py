"""Learning Loss strategy for RoMa active learning."""

from __future__ import annotations

import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

import roma
from roma.strategies.loss_prediction_module import LossPredictionModule
from roma.strategies.strategy_utils import log_strategy_action
from roma.utils.utils import to_cuda


DECODER_FEATURE_LAYER = "decoder.embedding_decoder.blocks"


def find_decoder_penultimate_layer(model) -> str:
    """Print decoder-related module names and return the hook target path."""
    base_model = model.module if hasattr(model, "module") else model
    for name, module in base_model.named_modules():
        if "decoder" in name.lower():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {type(module).__name__}, params={params}", flush=True)
    return DECODER_FEATURE_LAYER


def _resolve_module(root, dotted_name: str):
    module = root
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def get_decoder_features(model, batch) -> torch.Tensor:
    """Extract mean-pooled decoder token features before the gm_cls projection."""
    base_model = _base_model(model)
    captured = {}

    def hook_fn(_module, _inputs, output):
        captured["features"] = output

    try:
        target_layer = _resolve_module(base_model, DECODER_FEATURE_LAYER)
    except AttributeError as exc:
        raise RuntimeError(
            f"Could not resolve decoder feature hook target '{DECODER_FEATURE_LAYER}'. "
            "Run find_decoder_penultimate_layer(model) to inspect available modules."
        ) from exc

    handle = target_layer.register_forward_hook(hook_fn)
    try:
        _ = model(batch)
    finally:
        handle.remove()

    if "features" not in captured:
        raise RuntimeError(
            f"Hook did not capture decoder features from '{DECODER_FEATURE_LAYER}'."
        )

    feats = captured["features"]
    if feats.dim() == 3:
        feats = feats.mean(dim=1)
    elif feats.dim() == 4:
        feats = feats.mean(dim=(1, 2))
    elif feats.dim() != 2:
        raise RuntimeError(f"Unexpected decoder feature shape: {tuple(feats.shape)}")
    return feats.float()


def build_lpm_for_model(model) -> LossPredictionModule:
    """Create an LPM sized to the decoder hidden dimension of the given model."""
    base_model = _base_model(model)
    input_dim = int(base_model.decoder.embedding_decoder.hidden_dim)
    return LossPredictionModule(input_dim=input_dim)


def train_step_learn_loss(
    train_batch,
    model,
    objective,
    optimizer_roma,
    grad_scaler,
    lpm,
    optimizer_lpm,
    grad_clip_norm: float = 1.0,
    lambda_lpm: float = 1.0,
    margin: float = 1.0,
):
    """Train RoMa and the loss prediction module for one optimization step."""
    optimizer_roma.zero_grad()
    optimizer_lpm.zero_grad()

    corresps = model(train_batch)
    roma_loss, per_sample_loss = objective.forward_with_per_sample(corresps, train_batch)
    decoder_feats = get_decoder_features(model, train_batch)
    pred_losses = lpm(decoder_feats).squeeze(1)
    rank_loss = LossPredictionModule.ranking_loss(
        pred_losses, per_sample_loss.detach(), margin=margin
    )
    total_loss = roma_loss + lambda_lpm * rank_loss

    grad_scaler.scale(total_loss).backward()
    grad_scaler.unscale_(optimizer_roma)
    grad_scaler.unscale_(optimizer_lpm)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    torch.nn.utils.clip_grad_norm_(lpm.parameters(), grad_clip_norm)
    grad_scaler.step(optimizer_roma)
    grad_scaler.step(optimizer_lpm)
    grad_scaler.update()

    if grad_scaler._scale < 1.0:
        grad_scaler._scale = torch.tensor(1.0).to(grad_scaler._scale)
    roma.GLOBAL_STEP = roma.GLOBAL_STEP + roma.STEP_SIZE
    return {
        "train_out": corresps,
        "train_loss": float(total_loss.item()),
        "roma_loss": float(roma_loss.item()),
        "rank_loss": float(rank_loss.item()),
    }


def train_k_steps_learn_loss(
    n_0,
    k,
    dataloader,
    model,
    objective,
    optimizer_roma,
    optimizer_lpm,
    lr_scheduler,
    grad_scaler,
    lpm,
    grad_clip_norm: float = 1.0,
    lambda_lpm: float = 1.0,
    margin: float = 1.0,
):
    """Run ``k`` training steps for RoMa + LPM over an iterator of labeled batches."""
    model.train(True)
    lpm.train(True)
    for _ in range(n_0, n_0 + k):
        batch = next(dataloader)
        batch = to_cuda(batch)
        train_step_learn_loss(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer_roma=optimizer_roma,
            grad_scaler=grad_scaler,
            lpm=lpm,
            optimizer_lpm=optimizer_lpm,
            grad_clip_norm=grad_clip_norm,
            lambda_lpm=lambda_lpm,
            margin=margin,
        )
        lr_scheduler.step()


def score_unlabeled_pool(
    model,
    lpm: LossPredictionModule,
    avail: np.ndarray,
    strategy,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Score unlabeled pairs by predicted training loss."""
    dataset = strategy._entropy_dataset(avail)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.eval()
    lpm.eval()

    predicted_losses = []
    valid_ids = []
    with torch.no_grad():
        for pair_id, batch in zip(avail.astype(int).tolist(), dataloader):
            try:
                batch = {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                decoder_feats = get_decoder_features(model, batch)
                pred_loss = float(lpm(decoder_feats).reshape(-1)[0].item())
                predicted_losses.append(pred_loss)
                valid_ids.append(int(pair_id))
            except Exception as exc:
                log_strategy_action(
                    f"Learn loss: skipping pair {pair_id} during scoring ({exc})"
                )

    return (
        np.asarray(valid_ids, dtype=int),
        np.asarray(predicted_losses, dtype=np.float32),
    )


def load_lpm_for_cycle(model, checkpoint_root: str, stem: str, device) -> LossPredictionModule | None:
    """Load a persisted LPM checkpoint for selection, if present."""
    lpm_path = osp.join(checkpoint_root, f"{stem}_lpm_best.pth")
    if not osp.isfile(lpm_path):
        return None
    device = torch.device(device) if not isinstance(device, torch.device) else device
    lpm = build_lpm_for_model(model).to(device)
    state = torch.load(lpm_path, map_location=device)
    lpm.load_state_dict(state, strict=True)
    lpm.eval()
    return lpm


def save_lpm_checkpoint(lpm, checkpoint_root: str, stem: str) -> str:
    """Persist the best LPM checkpoint for use in the next AL cycle."""
    lpm_path = osp.join(checkpoint_root, f"{stem}_lpm_best.pth")
    torch.save(_base_model(lpm).state_dict(), lpm_path)
    return lpm_path


def run(strategy, k: int, model) -> np.ndarray:
    """Select top-k unlabeled samples with the highest predicted training loss."""
    avail = strategy.remaining()
    if avail.size == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(int(k), avail.size)

    lpm = getattr(strategy, "lpm", None)
    if lpm is None:
        log_strategy_action(
            "Learn loss: LPM not found on strategy. Falling back to random selection."
        )
        return strategy.rng.choice(avail, size=k, replace=False).astype(int)

    device = next(model.parameters()).device
    sample_ids, pred_losses = score_unlabeled_pool(
        model, lpm, avail, strategy, device=str(device)
    )
    if sample_ids.size == 0:
        return np.empty(0, dtype=int)

    order = np.argsort(pred_losses)[::-1]
    selected = sample_ids[order[:k]].astype(int)
    log_strategy_action(
        f"Learn loss: scored {len(sample_ids)} pairs, selected {selected.size} with highest predicted loss. "
        f"Pred loss range: [{pred_losses.min():.3f}, {pred_losses.max():.3f}]"
    )
    return selected
