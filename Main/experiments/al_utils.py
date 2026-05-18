import os
import os.path as osp
import numpy as np
import torch
# import torch.distributed as dist
# import cv2
import wandb
from roma.benchmarks import OpticalmapHomogBenchmark
from roma.benchmarks.metu_vistir_flat_benchmark import METUVisTIRFlatBenchmark
# from roma.utils import pose_auc

# EVAL_SEED = 42

DATASET_DIRS = {
    "opticalmap": "cross_modality/Optical-Map_12feb",
    "Optical-Map": "cross_modality/Optical-Map",
    "Optical-Infrared": "cross_modality/Optical-Infrared",
    "Optical-Depth": "cross_modality/Optical-Depth",
    "Optical-Optical": "cross_modality/Optical-Optical",
    "Nighttime": "cross_modality/Nighttime",
    "Map-Data": "cross_modality/Map-Data",
    "Optical-SAR": "cross_modality/Optical-SAR",
    "NIR-RGB": "cross_modality/NIR-RGB",
    "DPDN": "cross_modality/DPDN",
    "MSRS": "cross_modality/MSRS",
    "METU_VisTIR": "cross_modality/METU_VisTIR",
    "METU_Vis_TIR": "cross_modality/METU_VisTIR",
}

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def log_action(message: str):
    if is_rank0():
        print(f"[ACTION] {message}", flush=True)


def get_dataset_root(base_root, dataset_name):
    if dataset_name not in DATASET_DIRS:
        raise ValueError(dataset_name)
    return osp.join(base_root, DATASET_DIRS[dataset_name])


def load_model_weights(path, device_id):
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=f"cuda:{device_id}")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unexpected checkpoint format at {path}: type {type(ckpt)}")


def resolve_cycle0_seed_checkpoint(args, project_root="/projects/roma"):
    """Resolve the checkpoint used to seed cycle-0 selection/training."""
    preseed_job = f"{args.dataset_name}_preseed"
    return osp.join(project_root, args.dataset_name, preseed_job, f"{preseed_job}_cycle0_best.pth")


def _wandb_config(args, cycle, resolved_seed_path=None):
    config = dict(vars(args))
    config.update(
        {
            "cycle": int(cycle),
            "resolved_cycle0_seed_checkpoint": resolved_seed_path,
            "checkpoint_root": osp.join("/projects/_hdd/roma", args.dataset_name, args.job_name),
        }
    )
    return config


_FUNDAMENTAL_MATRIX_DATASETS = {"METU_VisTIR", "METU_Vis_TIR"}

def create_benchmarks(root, train_split, val_split, test_split, dataset_name=None):
    if dataset_name in _FUNDAMENTAL_MATRIX_DATASETS:
        cls = METUVisTIRFlatBenchmark
    else:
        cls = OpticalmapHomogBenchmark
    return (
        cls(root, train_split),
        cls(root, val_split),
        cls(root, test_split),
    )


# def distributed_benchmark(bench, model, world_size, rank):
#     """Shard benchmark across ranks, gather raw distances, return aggregated metrics."""
#     local_idx = bench.test_idx[rank::world_size]
#     torch.manual_seed(EVAL_SEED)
#     np.random.seed(EVAL_SEED)
#     cv2.setRNGSeed(EVAL_SEED)
#     homog_dists, all_epe = bench.benchmark_raw(model, idx_subset=local_idx)
#     if world_size > 1 and dist.is_initialized():
#         gathered = [None] * world_size
#         dist.all_gather_object(gathered, (homog_dists, all_epe))
#         all_homog = [d for hd, _ in gathered for d in hd]
#         all_epe_flat = [e for _, ep in gathered for e in ep]
#     else:
#         all_homog = homog_dists
#         all_epe_flat = all_epe
#     thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     auc = pose_auc(np.array(all_homog), thresholds)
#     return {"auc_3": auc[2], "auc_5": auc[4], "auc_10": auc[9], "epe": float(np.mean(all_epe_flat))}


def setup_wandb_run(args, cycle):
    if wandb.run is not None:
        wandb.finish()
    for var in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP"):
        os.environ.pop(var, None)
    mode = "online" if (not args.dont_log_wandb and is_rank0()) else "disabled"
    strategy = getattr(args, "strategy", "unknown")
    resolved_seed_path = resolve_cycle0_seed_checkpoint(args)
    wandb.init(
        project=f'ACCV_best_{args.dataset_name}_dataset',
        entity=args.wandb_entity,
        name=f"{args.job_name}/{strategy}/cycle_{cycle}",
        group=args.job_name,
        job_type=f"cycle_{cycle}",
        tags=[args.dataset_name, strategy, f"cycle_{cycle}", getattr(args, "train_resolution", "unknown")],
        config=_wandb_config(args, cycle, resolved_seed_path=resolved_seed_path),
        mode=mode,
        resume="never",
    )
    wandb.define_metric("step/global")
    wandb.define_metric("*", step_metric="step/global")


def close_wandb_run():
    if wandb.run is not None:
        wandb.finish()


def log_to_wandb(payload, step=None):
    if wandb.run is not None:
        wandb.log(payload, step=step)


def benchmark_metrics(prefix, metrics):
    return {
        f"{prefix}/auc_10": metrics.get("auc_10"),
        f"{prefix}/auc_5": metrics.get("auc_5"),
        f"{prefix}/auc_3": metrics.get("auc_3"),
        f"{prefix}/auc_20": metrics.get("auc_20"),
        f"{prefix}/epe": metrics.get("epe"),
        f"{prefix}/mean_pose_error": metrics.get("mean_pose_error"),
    }


def al_metrics(cycle, global_step, cycle_step=None, **values):
    payload = {
        "step/global": int(global_step),
        "step/cycle": int(cycle),
    }
    if cycle_step is not None:
        payload["step/cycle_step"] = int(cycle_step)
    for key, value in values.items():
        payload[f"al/{key}"] = value
    return payload




def update_checkpoints(checkpointer, model, optimizer, lr_scheduler, step, acc, acc_best):
    if acc > acc_best:
        acc_best = acc
        checkpointer.save_best(model, optimizer, lr_scheduler, step)
    checkpointer.save(model, optimizer, lr_scheduler, step)
    return acc_best
