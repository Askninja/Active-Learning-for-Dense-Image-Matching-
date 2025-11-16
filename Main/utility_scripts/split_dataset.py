import numpy as np
from pathlib import Path

SEED = 110105
COUNTS = (150, 20, 20, 10)
NUM_SAMPLES = sum(COUNTS)
DATA_ROOT = Path(__file__).resolve().parents[2] / 'datasets' / 'cross_modality' / 'Nighttime' / 'Idx_files'


def main() -> None:
    rng = np.random.default_rng(SEED)
    total = rng.permutation(np.arange(1, NUM_SAMPLES + 1))

    train = total[:COUNTS[0]]
    test = total[COUNTS[0]:COUNTS[0] + COUNTS[1]]
    val = total[COUNTS[0] + COUNTS[1]:COUNTS[0] + COUNTS[1] + COUNTS[2]]
    preseed = total[COUNTS[0] + COUNTS[1] + COUNTS[2]:COUNTS[0] + COUNTS[1] + COUNTS[2] + COUNTS[3]]

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(DATA_ROOT / 'train_idx.npy', train)
    np.save(DATA_ROOT / 'test_idx.npy', test)
    np.save(DATA_ROOT / 'val_idx.npy', val)
    np.save(DATA_ROOT / 'preseed_idx.npy', preseed)

    print(train.size, test.size, val.size, preseed.size)


if __name__ == '__main__':
    main()
