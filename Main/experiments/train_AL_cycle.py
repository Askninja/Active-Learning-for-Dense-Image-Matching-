import os
import os.path as osp
import numpy as np
from argparse import ArgumentParser
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
import wandb
from tqdm import tqdm
import roma
from experiments.train_roma_outdoor import get_model
from roma.datasets import OpticalMap
from roma.benchmarks import OpticalmapHomogBenchmark
from roma.losses.robust_loss import RobustLossesAMD, RobustLossesSymmetric
from roma.train.train import train_k_steps
from roma.checkpointing import CheckPoint
from roma.strategies.strategies import ActiveLearningStrategy
from experiments.al_utils import (
    is_rank0,
    log_action,
    get_dataset_root,
    load_model_weights,
    create_benchmarks,
    setup_wandb_run,
    close_wandb_run,
    log_to_wandb,
    update_checkpoints,
)


RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

DATASET_DIRS = {
    "opticalmap": "cross_modality/Optical-Map_12feb",
    "Optical-Map": "cross_modality/Optical-Map",
    "Optical-Infrared": "cross_modality/Optical-Infrared",
    "Optical-Depth": "cross_modality/Optical-Depth",
    "Optical-Optical": "cross_modality/Optical-Optical",
    "Nighttime": "cross_modality/Nighttime",
    "Map-Data": "cross_modality/Map-Data"
}



def train_active_learning(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device_id = local_rank
    roma.LOCAL_RANK = device_id
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    checkpoint_root = os.path.join("/projects/_hdd/roma", args.dataset_name, args.job_name)
    selector_seed_job = getattr(args, "selector_seed_job", None)
    if args.selector_seed_path is not None:
        default_seed_path = args.selector_seed_path
    elif selector_seed_job is None:
        selector_seed = "pretrained_seed"
        default_seed_path = osp.join(
            "/home/abhiram001/Active-Learning-for-Dense-Image-Matching-/Main/workspace/checkpoints",
            args.dataset_name,
            f"{selector_seed}.pth",
        )
    else:
       default_seed_path = osp.join(
            checkpoint_root,
            selector_seed_job,
            f"{selector_seed_job}_cycle0_best.pth"
        )

    log_action(f"Initialized distributed context (world_size={world_size}, rank={rank}, device={device_id}).")
    os.makedirs(checkpoint_root, exist_ok=True)
    h, w = RESOLUTIONS[args.train_resolution]
    roma.STEP_SIZE = world_size * args.gpu_batch_size
    N = int(args.N)
    k = max(1, int(args.eval_interval) // roma.STEP_SIZE)
    use_horizontal_flip_aug = "F" in args.aug
    use_cropping_aug = "C" in args.aug
    use_color_jitter_aug = "J" in args.aug
    use_swap_aug = "S" in args.aug
    use_dual_cropping_aug = "D" in args.aug
    symmetric = str(args.symmetric) in ("True", "true", "1")
    depth_interpolation_mode = "bilinear"
    needs_selector = args.strategy not in ("preseed", "full", "random")
    start_cycle = max(0, int(getattr(args, "start_cycle", 0)))
    
    
    
    for cycle in range(start_cycle, args.cycles):
        log_action(f"[cycle {cycle}] setup started.")
        setup_wandb_run(args, cycle)
        roma.GLOBAL_STEP = 0
        data_root = get_dataset_root(args.data_root, args.dataset_name)
        idx_root = osp.join(data_root, "Idx_files")
        train_split = f"train_{args.split}"
        val_split = f"val_{args.split}"
        test_split = f"test_{args.split}"
        log_action(f"[cycle {cycle}] Preparing selector for dataset at {data_root}.")
        selector = ActiveLearningStrategy(
            args,
            cycle,
            data_root=data_root,
            split=train_split,
            idx_root=idx_root,
        )
        stem = f"{args.job_name}_cycle{cycle}"
        selected_npy = osp.join(idx_root, f"{stem}.npy")
        train_split_path = f"Idx_files/{selector.split}"
        val_split_path = f"Idx_files/{val_split}"
        test_split_path = f"Idx_files/{test_split}"
        
        if is_rank0():
            sel_model = None
            if needs_selector:
                log_action(f"[cycle {cycle}] Loading selector model for strategy {args.strategy}.")
                sel_model = get_model(
                    pretrained_backbone=True,
                    resolution=args.train_resolution,
                    attenuate_cert=False,
                    symmetric=False,
                ).to(device_id)
                if cycle == 0:
                    selector_pretrained = default_seed_path
                    log_action(f"[cycle {cycle}] Loading selector seed checkpoint from {selector_pretrained}.")
                    sel_weights = load_model_weights(selector_pretrained, device_id)
                else:
                    prev_best = osp.join(
                        checkpoint_root,
                        f"{args.job_name}_cycle{cycle-1}_best.pth",
                    )
                    sel_weights = load_model_weights(prev_best, device_id)
                if cycle == 0:
                    log_action(f"[cycle {cycle}] Selector seed checkpoint loaded successfully.")
                else:
                    log_action(f"[cycle {cycle}] Loading selector weights from {prev_best}.")
                sel_model.load_state_dict(sel_weights, strict=True)
                sel_model.eval()
                log_action(f"[cycle {cycle}] Running selector to pick new indices.")
                selector.get_train_idx(model_for_uncertainty=sel_model)
                del sel_model
                torch.cuda.empty_cache()
            else:
                log_action(f"[cycle {cycle}] Running selector without model for strategy {args.strategy}.")
                selector.get_train_idx()
        if dist.is_initialized():
            dist.barrier()
        train_idx = np.load(selected_npy).astype(int)
        log_action(f"[cycle {cycle}] Loaded {train_idx.size} training indices from {selected_npy}.")
        target_train = OpticalMap(
            data_root=data_root,
            ht=h,
            wt=w,
            use_horizontal_flip_aug=use_horizontal_flip_aug,
            use_cropping_aug=use_cropping_aug,
            min_crop_ratio=args.min_crop_ratio,
            use_color_jitter_aug=use_color_jitter_aug,
            use_swap_aug=use_swap_aug,
            use_dual_cropping_aug=use_dual_cropping_aug,
            split=train_split_path,
        )
        target_train.train_idx = train_idx
        log_action(f"[cycle {cycle}] Building benchmarks for splits ({train_split_path}, {val_split_path}, {test_split_path}).")
        benchmark_train, benchmark_eval, benchmark_test = create_benchmarks(
            data_root, train_split_path, val_split_path, test_split_path
        )
        model = get_model(
            pretrained_backbone=True,
            resolution=args.train_resolution,
            attenuate_cert=False,
            symmetric=symmetric,
        ).to(device_id)
        if world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if not args.pretrained_path:
            raise ValueError("pretrained_path is required")
        
        weights = load_model_weights(args.pretrained_path, device_id)
        log_action(f"[cycle {cycle}] Loading model weights from {args.pretrained_path}.")
        model.load_state_dict(weights, strict=True)
        loss_class_target = RobustLossesSymmetric if symmetric else RobustLossesAMD
        log_action(f"[cycle {cycle}] Configuring target loss ({loss_class_target.__name__}).")
        depth_loss_target = loss_class_target(
            ce_weight=args.ce_weight,
            local_dist={1: 4, 2: 4, 4: 8, 8: 8},
            local_largest_scale=8,
            depth_interpolation_mode=depth_interpolation_mode,
            alpha=0.5,
            c=1e-4,
        )
        parameters = [
            {"params": model.encoder.parameters(), "lr": roma.STEP_SIZE * 5e-6 / 8},
            {"params": model.decoder.parameters(), "lr": roma.STEP_SIZE * args.dec_lr / 8},
        ]
        log_action(f"[cycle {cycle}] Creating optimizer and LR scheduler.")
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
        milestones = [int((9 * N / roma.STEP_SIZE) // 10)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        ddp_model = DDP(
            model,
            device_ids=[device_id],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        log_action(f"[cycle {cycle}] Wrapped model with DDP on device {device_id}.")
        grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
        grad_clip_norm = 0.01
        stem_ckpt = f"{args.job_name}_cycle{cycle}"
        checkpointer = CheckPoint(checkpoint_root, stem_ckpt)
        acc_best = float("-inf")
        log_action(f"[cycle {cycle}] Starting training chunks up to {N} global steps (chunk size {k}).")
        for n in range(roma.GLOBAL_STEP, N, k * roma.STEP_SIZE):
            total_chunk_samples = roma.STEP_SIZE * k
            if len(target_train) < roma.STEP_SIZE:
                raise ValueError(
                    f"Dataset has only {len(target_train)} samples, which is fewer than "
                    f"STEP_SIZE={roma.STEP_SIZE} (world_size × gpu_batch_size). "
                    f"Reduce --gpu_batch_size or use fewer GPUs."
                )
            chunk_seed = cycle * 1_000_000 + n
            generator = torch.Generator()
            generator.manual_seed(chunk_seed)
            # Sample with replacement when dataset is smaller than the chunk budget
            if len(target_train) >= total_chunk_samples:
                chunk_indices = torch.randperm(len(target_train), generator=generator)[:total_chunk_samples]
            else:
                chunk_indices = torch.randint(0, len(target_train), (total_chunk_samples,), generator=generator)
            local_chunk_indices = chunk_indices.view(world_size, args.gpu_batch_size * k)[rank].tolist()
            local_train_subset = Subset(target_train, local_chunk_indices)
            dataloader_target = iter(
                DataLoader(
                    local_train_subset,
                    batch_size=args.gpu_batch_size,
                    num_workers=world_size,
                    pin_memory=True,
                )
            )
            log_action(f"[cycle {cycle}] Training chunk starting at global step {n}.")
            train_k_steps(
                n,
                k,
                dataloader_target,
                ddp_model,
                depth_loss_target,
                optimizer,
                lr_scheduler,
                grad_scaler,
                grad_clip_norm=grad_clip_norm,
            )
            ddp_model.eval()
            with torch.no_grad():
                res_tr = benchmark_train.benchmark(ddp_model.module)
                res_ev = benchmark_eval.benchmark(ddp_model.module)
                res_te = benchmark_test.benchmark(ddp_model.module)
            if is_rank0():
                log_action(f"[cycle {cycle}] Logging metrics at global step {roma.GLOBAL_STEP}.")
                auc10_tr = float(res_tr.get("auc_10"))
                auc10_ev = float(res_ev.get("auc_10"))
                auc5_ev = float(res_ev.get("auc_5"))
                log_to_wandb(
                    {
                        "auc_10_train": auc10_tr,
                        "auc_10_val": auc10_ev,
                        "auc_5_train": res_tr.get("auc_5"),
                        "auc_5_val": res_ev.get("auc_5"),
                        "auc_3_train": res_tr.get("auc_3"),
                        "auc_3_val": res_ev.get("auc_3"),
                        "epe_train": res_tr.get("epe"),
                        "epe_val": res_ev.get("epe"),
                        "auc_10_test_current": res_te.get("auc_10"),
                        "auc_5_test_current": res_te.get("auc_5"),
                        "auc_3_test_current": res_te.get("auc_3"),
                        "epe_test_current": res_te.get("epe"),
                        "global_step": int(roma.GLOBAL_STEP),
                    }
                )
                acc = auc5_ev
                acc_best = update_checkpoints(
                    checkpointer,
                    ddp_model.module,
                    optimizer,
                    lr_scheduler,
                    roma.GLOBAL_STEP,
                    acc,
                    acc_best,
                )
                log_action(f"[cycle {cycle}] Checkpoints updated (best AUC5={acc_best:.4f}).")
            ddp_model.train()
            if dist.is_initialized():
                dist.barrier()
        if is_rank0():
            best_ckpt_path = f"/projects/_hdd/roma/{args.dataset_name}/{args.job_name}/{args.job_name}_cycle{cycle}_best.pth"
            if not osp.isfile(best_ckpt_path):
                raise FileNotFoundError(best_ckpt_path)
            best_states = torch.load(best_ckpt_path, map_location=f"cuda:{device_id}")
            if not (isinstance(best_states, dict) and "model" in best_states):
                raise RuntimeError(f"Checkpoint {best_ckpt_path} must be a dict with key 'model'.")
            ddp_model.module.load_state_dict(best_states["model"], strict=True)
            ddp_model.eval()
            with torch.no_grad():
                res_test = benchmark_test.benchmark(ddp_model.module)
            log_action(f"[cycle {cycle}] Evaluated on test set with restored best checkpoint.")
            log_to_wandb(
                {
                    "auc_10_test": res_test.get("auc_10"),
                    "auc_5_test": res_test.get("auc_5"),
                    "auc_3_test": res_test.get("auc_3"),
                    "epe_test": res_test.get("epe"),
                }
            )
        close_wandb_run()
        del ddp_model, model, optimizer, lr_scheduler, target_train
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        if is_rank0():
            log_action(f"[cycle {cycle}] completed successfully.")
            tqdm.write(f"[cycle {cycle}] completed.")
    if is_rank0():
        log_action("All active-learning cycles completed.")
        tqdm.write("All AL cycles completed.")
    if dist.is_initialized():
        dist.destroy_process_group()

def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--train_resolution", default="low")
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--wandb_entity", required=False)
    parser.add_argument(
        "--data_root",
        default="/home/abhiram001/active_learning/abhiram/AMD_ab/datasets/",
    )
    parser.add_argument("--job_name", default="opticalmap_split_0.8_train_coreset")
    parser.add_argument("--dataset_name", default="opticalmap")
    parser.add_argument("--pretrained_path", default="workspace/checkpoints/roma_outdoor.pth")
    parser.add_argument("--N", default=int(8e2), type=int)
    parser.add_argument("--eval_interval", default=5000, type=int, help="Global steps between each benchmark evaluation.")
    parser.add_argument("--ce_weight", default=0.01, type=float)
    parser.add_argument("--aug", default="F")
    parser.add_argument("--min_crop_ratio", default=0.5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--split", default="idx")
    parser.add_argument("--symmetric", default="False")
    parser.add_argument("--cycles", default=4, type=int)
    parser.add_argument("--start_cycle", default=0, type=int, help="Skip cycles before this index.")
    parser.add_argument("--selector_seed_path", default=None, help="Explicit path to selector checkpoint for cycle 0.")
    parser.add_argument("--selector_seed_job", default=None, help="Name of the job subfolder used to locate selector seed.")
    parser.add_argument(
        "--strategy",
        default="coreset",
        choices=[
            "preseed",
            "full",
            "random",
            "coreset",
            "geometry_diversity",
            "entropy_weighted_coreset",
            "hs_cert_weighted_coreset",
            "coreset2",
            "uncertainty",
            "kcenter_uncertainty_embedding",
            "kcenter_uncertainty_weighted_raw",
            "k_center_greedy_uncertainty",
            "entropy",
            "hs_cert",
            "coreset_appearance",
            "eigenvalue_diversity",
            "displacement_diversity",
            "combined_eigen_displacement",
            "hs_cert_weighted_eigenvalue_diversity",
        ],
    )
    parser.add_argument("--selector_batch_size", default=8, type=int)
    parser.add_argument("--geometry_hist_bins", default=16, type=int)
    parser.add_argument("--geometry_conf_threshold", default=0.5, type=float)
    parser.add_argument("--geometry_chunk_size", default=2048, type=int)
    return parser


if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    torch.backends.cudnn.allow_tf32 = True
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()
    roma.DEBUG_MODE = False
    train_active_learning(args)
