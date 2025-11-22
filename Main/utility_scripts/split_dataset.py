import numpy as np
from pathlib import Path

SEED = 110105
COUNTS = (150, 20, 20, 10)
NUM_SAMPLES = sum(COUNTS)
DATA_ROOT = Path(__file__).resolve().parents[2] / 'datasets' / 'cross_modality' / 'Optical-Optical' / 'Idx_files'


def main() -> None:
    rng = np.random.default_rng(SEED)
    total = rng.permutation(np.arange(1, NUM_SAMPLES + 1))
    preseed_path = DATA_ROOT / 'preseed_idx.npy'

    if preseed_path.is_file():
        # Reuse the preseed split so previously trained checkpoints stay aligned.
        preseed = np.load(preseed_path).astype(int)
        if preseed.size != COUNTS[3]:
            raise ValueError(f"Existing preseed has {preseed.size} samples; expected {COUNTS[3]}")
    else:
        preseed = total[-COUNTS[3]:]

    remaining = total[~np.isin(total, preseed)]
    expected_remaining = sum(COUNTS[:3])
    if remaining.size != expected_remaining:
        raise ValueError(f"Expected {expected_remaining} remaining samples, got {remaining.size}")

    train = remaining[:COUNTS[0]]
    test = remaining[COUNTS[0]:COUNTS[0] + COUNTS[1]]
    val = remaining[COUNTS[0] + COUNTS[1]:COUNTS[0] + COUNTS[1] + COUNTS[2]]

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(DATA_ROOT / 'train_idx.npy', train)
    np.save(DATA_ROOT / 'test_idx.npy', test)
    np.save(DATA_ROOT / 'val_idx.npy', val)
    np.save(DATA_ROOT / 'preseed_idx.npy', preseed)

    print(train.size, test.size, val.size, preseed.size)


if __name__ == '__main__':
    main()
