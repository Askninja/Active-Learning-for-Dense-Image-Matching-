import os
import os.path as osp
import torch
import wandb
from roma.benchmarks import OpticalmapHomogBenchmark

DATASET_DIRS = {
    "opticalmap": "cross_modality/Optical-Map_12feb",
    "Optical-Map": "cross_modality/Optical-Map",
    "Optical-Infrared": "cross_modality/Optical-Infrared",
    "Optical-Depth": "cross_modality/Optical-Depth",
    "Optical-Optical": "cross_modality/Optical-Optical",
    "Nighttime": "cross_modality/Nighttime",
    "Map-Data": "cross_modality/Map-Data"
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


def create_benchmarks(root, train_split, val_split, test_split):
    return (
        OpticalmapHomogBenchmark(root, train_split),
        OpticalmapHomogBenchmark(root, val_split),
        OpticalmapHomogBenchmark(root, test_split),
    )


def setup_wandb_run(args, cycle):
    if wandb.run is not None:
        wandb.finish()
    for var in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP"):
        os.environ.pop(var, None)
    mode = "online" if (not args.dont_log_wandb and is_rank0()) else "disabled"
    wandb.init(
        project=f'ACCV_{args.dataset_name}_dataset',
        entity=args.wandb_entity,
        name=f"{args.job_name}_cycle{cycle}",
        mode=mode,
        resume="never",
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")


def close_wandb_run():
    if wandb.run is not None:
        wandb.finish()


def log_to_wandb(payload):
    if wandb.run is not None:
        wandb.log(payload)


def update_checkpoints(checkpointer, model, optimizer, lr_scheduler, step, acc, acc_best):
    if acc > acc_best:
        acc_best = acc
        checkpointer.save_best(model, optimizer, lr_scheduler, step)
    checkpointer.save(model, optimizer, lr_scheduler, step)
    return acc_best