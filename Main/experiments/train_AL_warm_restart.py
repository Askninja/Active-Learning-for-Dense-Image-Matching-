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
    distributed_benchmark,
    log_to_wandb,
    update_checkpoints,
    resolve_cycle0_seed_checkpoint,
    benchmark_metrics,
    al_metrics,
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
    "Optical-SAR": "cross_modality/Optical-SAR",
}


def _setup_wandb(args):
    if wandb.run is not None:
        wandb.finish()
    mode = "online" if (not args.dont_log_wandb and is_rank0()) else "disabled"
    strategy = getattr(args, "strategy", "unknown")
    wandb.init(
        project=f"roma_active_learning_{args.dataset_name}",
        entity=args.wandb_entity,
        name=f"{args.job_name}/{strategy}/warm_restart",
        group=args.job_name,
        job_type="warm_restart",
        tags=[args.dataset_name, strategy, "warm_restart", getattr(args, "train_resolution", "unknown")],
        config=dict(vars(args)),
        mode=mode,
        resume="never",
    )
    wandb.define_metric("step/global")
    wandb.define_metric("*", step_metric="step/global")


def train_active_learning_warm_restart(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device_id = local_rank
    roma.LOCAL_RANK = device_id
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    checkpoint_root = os.path.join("/projects/_hdd/roma", args.dataset_name, args.job_name)
    default_seed_path = resolve_cycle0_seed_checkpoint(args)

    log_action(f"Initialized distributed context (world_size={world_size}, rank={rank}, device={device_id}).")
    os.makedirs(checkpoint_root, exist_ok=True)

    h, w = RESOLUTIONS[args.train_resolution]
    roma.STEP_SIZE = world_size * args.gpu_batch_size
    # k = number of optimizer steps per cycle (one eval_interval worth of training)
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

    # Single wandb run spanning all cycles; global_step is the x-axis.
    if is_rank0():
        _setup_wandb(args)

    # GLOBAL_STEP accumulates across cycles so wandb plots are continuous.
    roma.GLOBAL_STEP = 0

    for cycle in range(start_cycle, args.cycles):
        log_action(f"[cycle {cycle}] setup started.")

        data_root = get_dataset_root(args.data_root, args.dataset_name)
        idx_root = osp.join(data_root, "Idx_files")
        train_split = f"train_{args.split}"
        val_split = f"val_{args.split}"
        test_split = f"test_{args.split}"

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

        # --- Selection (rank-0 only) ---
        if is_rank0():
            sel_model = None
            if needs_selector:
                log_action(f"[cycle {cycle}] Loading selector model.")
                sel_model = get_model(
                    pretrained_backbone=True,
                    resolution=args.train_resolution,
                    attenuate_cert=False,
                    symmetric=False,
                ).to(device_id)
                if cycle == 0:
                    selector_pretrained = default_seed_path
                    log_action(f"[cycle {cycle}] Selector seed: {selector_pretrained}.")
                else:
                    selector_pretrained = osp.join(
                        checkpoint_root,
                        f"{args.job_name}_cycle{cycle - 1}_best.pth",
                    )
                    log_action(f"[cycle {cycle}] Selector weights from prev best: {selector_pretrained}.")
                sel_weights = load_model_weights(selector_pretrained, device_id)
                sel_model.load_state_dict(sel_weights, strict=True)
                sel_model.eval()
                selector.get_train_idx(model_for_uncertainty=sel_model)
                del sel_model
                torch.cuda.empty_cache()
            else:
                log_action(f"[cycle {cycle}] Running selector without model ({args.strategy}).")
                selector.get_train_idx()

        if dist.is_initialized():
            dist.barrier()

        train_idx = np.load(selected_npy).astype(int)
        log_action(f"[cycle {cycle}] Loaded {train_idx.size} training indices.")
        if is_rank0():
            log_to_wandb(
                al_metrics(
                    cycle,
                    roma.GLOBAL_STEP,
                    selected_count=int(train_idx.size),
                    pool_size=int(selector.train_pool_idx.size),
                    remaining_count=int(selector.remaining().size),
                    strategy=args.strategy,
                    selected_idx_path=selected_npy,
                )
            )

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

        benchmark_train, benchmark_eval, benchmark_test = create_benchmarks(
            data_root, train_split_path, val_split_path, test_split_path
        )

        # --- Model: warm restart ---
        model = get_model(
            pretrained_backbone=True,
            resolution=args.train_resolution,
            attenuate_cert=False,
            symmetric=symmetric,
        ).to(device_id)
        if world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Cycle 0 warms from the same seed as the selector (not a generic pretrained).
        # Cycle 1+ warms from the previous cycle's best checkpoint.
        if cycle == 0:
            warm_start_path = default_seed_path
        else:
            warm_start_path = osp.join(
                checkpoint_root,
                f"{args.job_name}_cycle{cycle - 1}_best.pth",
            )
        log_action(f"[cycle {cycle}] Warm restart from {warm_start_path}.")
        weights = load_model_weights(warm_start_path, device_id)
        model.load_state_dict(weights, strict=True)

        loss_class_target = RobustLossesSymmetric if symmetric else RobustLossesAMD
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
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
        # Scheduler milestone relative to steps within this cycle.
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(9 * k // 10)]
        )

        ddp_model = DDP(
            model,
            device_ids=[device_id],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
        grad_clip_norm = 0.01

        # --- Build dataloader for this cycle's eval_interval chunk ---
        total_chunk_samples = roma.STEP_SIZE * k
        if len(target_train) < roma.STEP_SIZE:
            raise ValueError(
                f"Dataset has only {len(target_train)} samples, fewer than "
                f"STEP_SIZE={roma.STEP_SIZE}. Reduce --gpu_batch_size or use fewer GPUs."
            )
        chunk_seed = cycle * 1_000_000
        generator = torch.Generator()
        generator.manual_seed(chunk_seed)
        if len(target_train) >= total_chunk_samples:
            chunk_indices = torch.randperm(len(target_train), generator=generator)[:total_chunk_samples]
        else:
            chunk_indices = torch.randint(
                0, len(target_train), (total_chunk_samples,), generator=generator
            )
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

        # --- Train for eval_interval steps ---
        n_start = roma.GLOBAL_STEP
        log_action(f"[cycle {cycle}] Training {k} steps from global step {n_start}.")
        train_k_steps(
            n_start,
            k,
            dataloader_target,
            ddp_model,
            depth_loss_target,
            optimizer,
            lr_scheduler,
            grad_scaler,
            grad_clip_norm=grad_clip_norm,
        )

        # --- Eval ---
        ddp_model.eval()
        with torch.no_grad():
            res_ev = distributed_benchmark(benchmark_eval, ddp_model.module, world_size, rank)
            res_te = distributed_benchmark(benchmark_test, ddp_model.module, world_size, rank)

        if is_rank0():
            log_action(f"[cycle {cycle}] Logging metrics at global step {roma.GLOBAL_STEP}.")
            auc5_ev = float(res_ev.get("auc_5"))
            payload = al_metrics(cycle, roma.GLOBAL_STEP, cycle_step=roma.GLOBAL_STEP - n_start)
            payload.update(benchmark_metrics("val", res_ev))
            payload.update(benchmark_metrics("test/current", res_te))
            log_to_wandb(payload)
            # acc_best starts at -inf per cycle so save_best always triggers (one eval/cycle).
            stem_ckpt = f"{args.job_name}_cycle{cycle}"
            checkpointer = CheckPoint(checkpoint_root, stem_ckpt)
            update_checkpoints(
                checkpointer,
                ddp_model.module,
                optimizer,
                lr_scheduler,
                roma.GLOBAL_STEP,
                auc5_ev,
                float("-inf"),
            )
            log_action(f"[cycle {cycle}] Checkpoint saved (AUC5={auc5_ev:.4f}).")

        ddp_model.train()
        if dist.is_initialized():
            dist.barrier()

        del ddp_model, model, optimizer, lr_scheduler, target_train
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        if is_rank0():
            log_action(f"[cycle {cycle}] completed.")
            tqdm.write(f"[cycle {cycle}] completed.")

    # Final test eval using the last cycle's best checkpoint.
    if is_rank0():
        last_cycle = args.cycles - 1
        best_ckpt_path = osp.join(
            checkpoint_root,
            f"{args.job_name}_cycle{last_cycle}_best.pth",
        )
        if osp.isfile(best_ckpt_path):
            final_model = get_model(
                pretrained_backbone=True,
                resolution=args.train_resolution,
                attenuate_cert=False,
                symmetric=symmetric,
            ).to(device_id)
            best_states = torch.load(best_ckpt_path, map_location=f"cuda:{device_id}")
            final_model.load_state_dict(best_states["model"], strict=True)
            final_model.eval()
            data_root = get_dataset_root(args.data_root, args.dataset_name)
            test_split_path = f"Idx_files/test_{args.split}"
            _, _, benchmark_test = create_benchmarks(
                data_root,
                f"Idx_files/train_{args.split}",
                f"Idx_files/val_{args.split}",
                test_split_path,
            )
            with torch.no_grad():
                res_test = benchmark_test.benchmark(final_model)
            payload = al_metrics(last_cycle, roma.GLOBAL_STEP, best_checkpoint_path=best_ckpt_path)
            payload.update(benchmark_metrics("test/best", res_test))
            log_to_wandb(payload)
            log_action("Final test evaluation logged.")
            del final_model

    if wandb.run is not None:
        wandb.finish()

    if is_rank0():
        log_action("All warm-restart AL cycles completed.")
        tqdm.write("All warm-restart AL cycles completed.")
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
    parser.add_argument("--job_name", default="opticalmap_warm_restart")
    parser.add_argument("--dataset_name", default="opticalmap")
    parser.add_argument("--selector_seed_path", default=None)
    parser.add_argument("--selector_seed_job", default=None)
    parser.add_argument(
        "--eval_interval",
        default=5000,
        type=int,
        help="Training steps per AL cycle (also the resampling interval).",
    )
    parser.add_argument("--ce_weight", default=0.01, type=float)
    parser.add_argument("--aug", default="F")
    parser.add_argument("--min_crop_ratio", default=0.5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--split", default="idx")
    parser.add_argument("--symmetric", default="False")
    parser.add_argument("--cycles", default=4, type=int, help="Number of AL resampling cycles.")
    parser.add_argument("--start_cycle", default=0, type=int)
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
            "entropy_weighted_geometric_diversity",
            "hs_cert_weighted_geometric_diversity",
            "hs_cert_delta4_geomdiv",
            "hs_cert_new",
            "hs_cert_3",
            "combined_diversity",
            "combined_metric_diversity",
            "uncertainty_metric_diversity",
            "badge",
            "learn_loss",
            "pairwise_hscert_df_dg",
            "pairwise_entropy_df_dg",
            "pairwise_hscert_dg",
            "pairwise_hscert_df",
            "w2_logdet",
            "alpha",
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
    train_active_learning_warm_restart(args)
