import os
import os.path as osp
import numpy as np
from argparse import ArgumentParser
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm
import roma
from experiments.train_roma_outdoor import get_model
from roma.datasets import AMD, OpticalMap
from roma.benchmarks import AmdHomogBenchmark, OpticalmapHomogBenchmark
from roma.losses.robust_loss import RobustLossesAMD, RobustLossesSymmetric
from roma.train.train import train_k_steps
from roma.checkpointing import CheckPoint
from roma.strategies.strategies import ActiveLearningStrategy

RESOLUTIONS = {
    "low": (448, 448),
    "medium": (14 * 8 * 5, 14 * 8 * 5),
    "high": (14 * 8 * 6, 14 * 8 * 6),
}

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def train_active_learning(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device_id = local_rank
    roma.LOCAL_RANK = device_id
    checkpoint_root = os.path.join("/projects/_hdd/roma", args.dataset_name, args.job_name)
    os.makedirs(checkpoint_root, exist_ok=True)
    h, w = RESOLUTIONS[args.train_resolution]
    roma.STEP_SIZE = world_size * args.gpu_batch_size
    N = int(args.N)
    k = max(1, 24996 // roma.STEP_SIZE)
    use_horizontal_flip_aug = "F" in args.aug
    use_cropping_aug = "C" in args.aug
    use_color_jitter_aug = "J" in args.aug
    use_swap_aug = "S" in args.aug
    use_dual_cropping_aug = "D" in args.aug
    symmetric = str(args.symmetric) in ("True", "true", "1")
    depth_interpolation_mode = "bilinear"
    needs_selector = args.strategy in (
        "random",
        "coreset",
        "coreset2",
        "roma_homography_stability",
        "kcenter_uncertainty_weighted",
        "adaptive_homog_uwe",
        "dpp",
        "weighted_dpp_tau1",
        "weighted_tau_dpp",
        "tau_weighted_embedding",
    )
    for cycle in range(args.cycles):
        if wandb.run is not None:
            wandb.finish()
        for v in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP"):
            os.environ.pop(v, None)
        wandb_mode = "online" if (not args.dont_log_wandb and is_rank0()) else "disabled"
        wandb.init(
            project=f"Roma_{args.dataset_name}",
            entity=args.wandb_entity,
            name=f"{args.job_name}_cycle{cycle}",
            mode=wandb_mode,
            resume="never",
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        roma.GLOBAL_STEP = 0
        if args.dataset_name == "opticalmap":
            data_root = osp.join(args.data_root, "cross_modality/Optical-Map_12feb")
        elif args.dataset_name == "Optical-Map":
            data_root = osp.join(args.data_root, "cross_modality/Optical-Map")
        elif args.dataset_name == "Optical-Infrared":
            data_root = osp.join(args.data_root, "cross_modality/Optical-Infrared")
        elif args.dataset_name == "Optical-Depth":
            data_root = osp.join(args.data_root, "cross_modality/Optical-Depth")
        elif args.dataset_name == "Optical-Optical":
            data_root = osp.join(args.data_root, "cross_modality/Optical-Optical")
        elif args.dataset_name == "Nighttime":
            data_root = osp.join(args.data_root, "cross_modality/Nighttime")
        else:
            raise ValueError(args.dataset_name)
        selector = ActiveLearningStrategy(args, cycle, data_root=data_root, split="train_" + args.split)
        stem = f"{args.job_name}_{'train_' + args.split}_cycle{cycle}_strategy_{args.strategy}"
        selected_npy = osp.join(data_root, f"{stem}.npy")
        if is_rank0():
            sel_model = None
            if needs_selector:
                sel_model = get_model(
                    pretrained_backbone=True,
                    resolution=args.train_resolution,
                    attenuate_cert=False,
                    symmetric=False,
                ).to(device_id)
                if cycle == 0:
                    selector_pretrained = f"workspace/checkpoints/{args.dataset_name}/pretrained_seed.pth"
                    if not osp.isfile(selector_pretrained):
                        raise FileNotFoundError(selector_pretrained)
                    ckpt = torch.load(selector_pretrained, map_location=f"cuda:{device_id}")
                    if isinstance(ckpt, dict) and "model" in ckpt:
                        sel_weights = ckpt["model"]
                    elif isinstance(ckpt, dict):
                        sel_weights = ckpt
                    else:
                        raise RuntimeError(
                            f"Unexpected checkpoint format at {selector_pretrained}: "
                            f"type {type(ckpt)}"
                        )
                    sel_model.load_state_dict(sel_weights, strict=True)
                else:
                    prev_best = osp.join(
                        checkpoint_root,
                        f"{args.job_name}_cycle{cycle-1}_strategy_{args.strategy}_best.pth",
                    )
                    if not osp.isfile(prev_best):
                        raise FileNotFoundError(prev_best)
                    ckpt = torch.load(prev_best, map_location=f"cuda:{device_id}")
                    if isinstance(ckpt, dict) and "model" in ckpt:
                        sel_weights = ckpt["model"]
                    elif isinstance(ckpt, dict):
                        sel_weights = ckpt
                    else:
                        raise RuntimeError(
                            f"Unexpected checkpoint format at {prev_best}: "
                            f"type {type(ckpt)}"
                        )
                    sel_model.load_state_dict(sel_weights, strict=True)
                sel_model.eval()
                selector.get_train_idx(model_for_uncertainty=sel_model)
                del sel_model
                torch.cuda.empty_cache()
            else:
                selector.get_train_idx()
            if cycle == 0:
                add_values = np.array([15, 37, 98, 114, 137, 141, 142, 182, 194, 195])
                arr = np.load(selected_npy)
                updated = np.unique(np.concatenate((arr, add_values)))
                np.save(selected_npy, updated)
        else:
            import time
            for _ in range(36000):
                if osp.isfile(selected_npy):
                    break
                time.sleep(1)
        train_idx = np.load(selected_npy).astype(int)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        data_root_om = osp.join(args.data_root, "cross_modality", args.dataset_name)
        target_train = OpticalMap(
            data_root=data_root_om,
            ht=h,
            wt=w,
            use_horizontal_flip_aug=use_horizontal_flip_aug,
            use_cropping_aug=use_cropping_aug,
            min_crop_ratio=args.min_crop_ratio,
            use_color_jitter_aug=use_color_jitter_aug,
            use_swap_aug=use_swap_aug,
            use_dual_cropping_aug=use_dual_cropping_aug,
            split=selector.split,
        )
        target_train.train_idx = train_idx
        benchmark_train = OpticalmapHomogBenchmark(data_root_om, selector.split)
        benchmark_eval = OpticalmapHomogBenchmark(data_root_om, "val_" + args.split)
        benchmark_test = OpticalmapHomogBenchmark(data_root_om, "test_" + args.split)
        model = get_model(
            pretrained_backbone=True,
            resolution=args.train_resolution,
            attenuate_cert=False,
            symmetric=symmetric,
        ).to(device_id)
        if not args.pretrained_path or not osp.isfile(args.pretrained_path):
            raise FileNotFoundError(str(args.pretrained_path))
        ckpt = torch.load(args.pretrained_path, map_location=f"cuda:{device_id}")
        if isinstance(ckpt, dict) and "model" in ckpt:
            weights = ckpt["model"]
        elif isinstance(ckpt, dict):
            weights = ckpt
        else:
            raise RuntimeError(
                f"Unexpected checkpoint format at {args.pretrained_path}: "
                f"type {type(ckpt)}"
            )
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
        milestones = [int((9 * N / roma.STEP_SIZE) // 10)]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
        ddp_model = DDP(
            model,
            device_ids=[device_id],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        grad_scaler = torch.cuda.amp.GradScaler(growth_interval=1_000_000)
        grad_clip_norm = 0.01
        stem_ckpt = f"{args.job_name}_cycle{cycle}_strategy_{args.strategy}"
        checkpointer = CheckPoint(checkpoint_root, stem_ckpt)
        acc_best = float("-inf")
        for n in range(roma.GLOBAL_STEP, N, k * roma.STEP_SIZE):
            sampler = RandomSampler(
                target_train,
                num_samples=args.gpu_batch_size * k,
                replacement=False,
            )
            dataloader_target = iter(
                DataLoader(
                    target_train,
                    batch_size=args.gpu_batch_size,
                    sampler=sampler,
                    num_workers=world_size,
                    pin_memory=True,
                )
            )
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
            if is_rank0():
                auc10_tr = float(res_tr.get("auc_10"))
                auc10_ev = float(res_ev.get("auc_10"))
                wandb.log(
                    {
                        "auc_10_train": auc10_tr,
                        "auc_10_val": auc10_ev,
                        "auc_5_train": res_tr.get("auc_5"),
                        "auc_5_val": res_ev.get("auc_5"),
                        "auc_3_train": res_tr.get("auc_3"),
                        "auc_3_val": res_ev.get("auc_3"),
                        "epe_train": res_tr.get("epe"),
                        "epe_val": res_ev.get("epe"),
                        "global_step": int(roma.GLOBAL_STEP),
                    }
                )
                acc = auc10_ev
                if acc > acc_best:
                    acc_best = acc
                    checkpointer.save_best(ddp_model.module, optimizer, lr_scheduler, roma.GLOBAL_STEP)
                checkpointer.save(ddp_model.module, optimizer, lr_scheduler, roma.GLOBAL_STEP)
            ddp_model.train()
            if dist.is_initialized():
                dist.barrier()
        if is_rank0():
            best_ckpt_path = f"/projects/_hdd/roma/{args.dataset_name}/{args.job_name}/{args.job_name}_cycle{cycle}_strategy_{args.strategy}_best.pth"
            if not osp.isfile(best_ckpt_path):
                raise FileNotFoundError(best_ckpt_path)
            best_states = torch.load(best_ckpt_path, map_location=f"cuda:{device_id}")
            if not (isinstance(best_states, dict) and "model" in best_states):
                raise RuntimeError(f"Checkpoint {best_ckpt_path} must be a dict with key 'model'.")
            ddp_model.module.load_state_dict(best_states["model"], strict=True)
            ddp_model.eval()
            with torch.no_grad():
                res_test = benchmark_test.benchmark(ddp_model.module)
            wandb.log(
                {
                    "auc_10_test": res_test.get("auc_10"),
                    "auc_5_test": res_test.get("auc_5"),
                    "auc_3_test": res_test.get("auc_3"),
                    "epe_test": res_test.get("epe"),
                }
            )
        if wandb.run is not None:
            wandb.finish()
        del ddp_model, model, optimizer, lr_scheduler, target_train
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()
        if is_rank0():
            tqdm.write(f"[cycle {cycle}] completed.")
    if is_rank0():
        tqdm.write("All AL cycles completed.")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    torch.backends.cudnn.allow_tf32 = True
    parser = ArgumentParser()
    parser.add_argument("--dont_log_wandb", action="store_true")
    parser.add_argument("--train_resolution", default="low")
    parser.add_argument("--gpu_batch_size", default=4, type=int)
    parser.add_argument("--wandb_entity", required=False)
    parser.add_argument("--data_root", default="/home/caill/datasets/")
    parser.add_argument("--job_name", default="opticalmap_split_0.8_train_coreset")
    parser.add_argument("--dataset_name", default="opticalmap")
    parser.add_argument("--pretrained_path", default="workspace/checkpoints/roma_outdoor.pth")
    parser.add_argument("--N", default=int(8e2), type=int)
    parser.add_argument("--ce_weight", default=0.01, type=float)
    parser.add_argument("--aug", default="F")
    parser.add_argument("--min_crop_ratio", default=0.5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--split", default="train")
    parser.add_argument("--symmetric", default="False")
    parser.add_argument("--cycles", default=4, type=int)
    parser.add_argument(
        "--strategy",
        default="coreset",
        choices=[
            "random",
            "coreset",
            "coreset2",
            "roma_homography_stability",
            "kcenter_uncertainty_weighted",
            "adaptive_homog_uwe",
            "dpp",
            "weighted_dpp_tau1",
            "weighted_tau_dpp",
            "tau_weighted_embedding",
        ],
    )
    args, _ = parser.parse_known_args()
    roma.DEBUG_MODE = False
    train_active_learning(args)
