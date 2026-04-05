# Active Learning for Dense Image Matching

Active learning framework built on top of [RoMa](https://github.com/Parskatt/RoMa) for cross-modality dense image matching. The system iteratively selects the most informative unlabeled image pairs to label, fine-tuning the matcher over multiple active learning cycles.

---

## Overview

Standard supervised fine-tuning of dense matchers requires large labeled datasets. This framework reduces annotation cost by using active learning: starting from a small seed set, at each cycle the model selects a budget of new pairs to label, fine-tunes on the growing labeled set, and repeats.

The framework supports multiple selection strategies ranging from pure uncertainty sampling to geometric diversity methods.

---

## Datasets

| Name | Description |
|---|---|
| `Optical-Depth` | Optical vs. depth modality pairs |
| `Optical-Infrared` | Optical vs. infrared pairs |
| `Optical-Optical` | Same-modality optical pairs |
| `Optical-Map` | Optical vs. map pairs |
| `Nighttime` | Day vs. nighttime pairs |
| `Map-Data` | Map data pairs |

---

## Active Learning Strategies

Each strategy has its own implementation file in `Main/roma/strategies/` and a corresponding SLURM script in `Main/slurmfiles/Optical-Depth/`.

### Baselines

| Strategy | File | Description |
|---|---|---|
| `random` | `strategy_random.py` | Uniform random sampling |
| `full` | — | Train on the entire pool (upper bound) |
| `preseed` | — | Fixed seed set only (lower bound) |

### Uncertainty-based

| Strategy | File | Description |
|---|---|---|
| `entropy` | `strategy_entropy.py` | Top-k by prediction entropy of the coarse GM classifier |
| `hs_cert` | `strategy_hs_cert.py` | Top-k by homography-spread uncertainty (RANSAC variance over match subsets) |

### Diversity-based (Appearance)

| Strategy | File | Description |
|---|---|---|
| `coreset` | `strategy_coreset.py` | KMeans + k-center greedy on RoMa backbone embeddings |
| `coreset_appearance` | `strategy_coreset_appearance.py` | Plain k-center greedy on L2-normalized backbone embeddings |

### Diversity-based (Geometry)

| Strategy | File | Description |
|---|---|---|
| `eigenvalue_diversity` | `strategy_eigenvalue_diversity.py` | K-center greedy on eigenvalue spectrum of mean RANSAC homography (6-d) |
| `displacement_diversity` | `strategy_displacement_diversity.py` | K-center greedy on corner displacement vectors of mean RANSAC homography (8-d) |
| `combined_eigen_displacement` | `strategy_combined_eigen_displacement.py` | K-center greedy on concatenated eigenvalue + displacement features (14-d) |
| `geometry_diversity` | `strategy_geometry_diversity.py` | Novelty scoring via flow magnitude/direction histograms vs. labeled set |

### Hybrid (Uncertainty + Diversity)

| Strategy | File | Description |
|---|---|---|
| `entropy_weighted_coreset` | `strategy_entropy_weighted_coreset.py` | Entropy-scaled appearance embeddings + KMeans k-center |
| `hs_cert_weighted_coreset` | `strategy_hs_cert_weighted_coreset.py` | HS-Cert-scaled appearance embeddings + KMeans k-center |
| `hs_cert_weighted_eigenvalue_diversity` | `strategy_hs_cert_weighted_eigenvalue_diversity.py` | HS-Cert-scaled eigenvalue features + k-center greedy |

---

## Project Structure

```
Main/
├── experiments/
│   ├── train_AL_cycle.py          # Main active learning training loop
│   └── train_roma_outdoor.py      # Base RoMa model builder
├── roma/
│   ├── strategies/
│   │   ├── strategies.py          # ActiveLearningStrategy class (infrastructure + dispatch)
│   │   ├── strategy_utils.py      # Shared utilities (k_center_greedy, homography/entropy/geometry helpers)
│   │   ├── strategy_random.py
│   │   ├── strategy_coreset.py
│   │   ├── strategy_entropy.py
│   │   ├── strategy_hs_cert.py
│   │   ├── strategy_entropy_weighted_coreset.py
│   │   ├── strategy_hs_cert_weighted_coreset.py
│   │   ├── strategy_geometry_diversity.py
│   │   ├── strategy_coreset_appearance.py
│   │   ├── strategy_eigenvalue_diversity.py
│   │   ├── strategy_displacement_diversity.py
│   │   ├── strategy_combined_eigen_displacement.py
│   │   └── strategy_hs_cert_weighted_eigenvalue_diversity.py
│   ├── models/                    # RoMa model architecture
│   ├── losses/                    # Robust loss functions
│   ├── benchmarks/                # Evaluation benchmarks
│   └── datasets/                  # Dataset loaders
├── slurmfiles/
│   └── Optical-Depth/             # One SLURM script per strategy
└── workspace/
    └── checkpoints/               # Pretrained and fine-tuned model weights
```

---

## Running Experiments

### SLURM

Each strategy has a ready-to-submit SLURM script:

```bash
sbatch Main/slurmfiles/Optical-Depth/<strategy_name>.sh
```

For example:

```bash
sbatch Main/slurmfiles/Optical-Depth/hs_cert_weighted_eigenvalue_diversity.sh
```

### Manual Launch

```bash
cd Main
python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    experiments/train_AL_cycle.py \
    --dataset_name=Optical-Depth \
    --data_root=/path/to/datasets/ \
    --job_name=my_experiment \
    --pretrained_path=workspace/checkpoints/roma_outdoor.pth \
    --strategy=hs_cert_weighted_eigenvalue_diversity \
    --cycles=6 \
    --N=50000 \
    --gpu_batch_size=3 \
    --train_resolution=medium
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--strategy` | `coreset` | Active learning selection strategy |
| `--cycles` | `4` | Number of AL cycles |
| `--N` | `800` | Training steps per cycle |
| `--eval_interval` | `5000` | Global steps between benchmark evaluations |
| `--gpu_batch_size` | `4` | Per-GPU batch size |
| `--train_resolution` | `low` | Image resolution (`low`, `medium`, `high`) |
| `--ce_weight` | `0.01` | Cross-entropy loss weight |
| `--dec_lr` | `1e-4` | Decoder learning rate |
| `--aug` | `F` | Augmentation mode (`F`, `FCSD`, etc.) |
| `--min_crop_ratio` | `0.5` | Minimum crop ratio for augmentation |
| `--split` | `idx` | Dataset split identifier |
| `--pretrained_path` | — | Path to pretrained RoMa checkpoint |

---

## Budget Schedule

The fraction of the unlabeled pool selected at each cycle:

| Cycle | Budget |
|---|---|
| 0 | 10% |
| 1 | 20% |
| 2 | 20% |
| 3+ | 30% |

---

## Evaluation

Metrics logged to [Weights & Biases](https://wandb.ai) at each `eval_interval`:

- `auc_10`, `auc_5`, `auc_3` — Area under the pose error curve at 10°/5°/3°
- `epe` — End-point error

Separate metrics are reported for the train, validation, and test splits at each cycle.
