import numpy as np
from pathlib import Path

# ── hard-coded parameters ─────────────────────────────────────────────────────
SEED         = 110105
TRAIN_SIZE   = 130
TEST_SIZE    = 30
VAL_SIZE     = 30
PRESEED_SIZE = 10

DATASET_NAME = "Optical-Optical"

NUM_SAMPLES  = TRAIN_SIZE + TEST_SIZE + VAL_SIZE + PRESEED_SIZE

DATA_ROOT = Path("/projects/ALData/Active_Datasets/cross_modality") / DATASET_NAME / "Idx_files"
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    rng   = np.random.default_rng(SEED)
    total = rng.permutation(np.arange(1, NUM_SAMPLES + 1))

    preseed = total[:PRESEED_SIZE]
    train   = total[PRESEED_SIZE : PRESEED_SIZE + TRAIN_SIZE]
    test    = total[PRESEED_SIZE + TRAIN_SIZE : PRESEED_SIZE + TRAIN_SIZE + TEST_SIZE]
    val     = total[PRESEED_SIZE + TRAIN_SIZE + TEST_SIZE :]

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    np.save(DATA_ROOT / "train.npy",   train)
    np.save(DATA_ROOT / "test.npy",    test)
    np.save(DATA_ROOT / "val.npy",     val)
    np.save(DATA_ROOT / "preseed.npy", preseed)

    print(f"train={train.size}  test={test.size}  val={val.size}  preseed={preseed.size}")


if __name__ == "__main__":
    main()