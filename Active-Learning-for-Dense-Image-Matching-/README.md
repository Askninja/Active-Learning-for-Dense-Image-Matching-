# Active-Learning-for-Dense-Image-Matching-

## Determinism

Training now exposes a repo-wide seed through `Main/experiments/train_AL_cycle.py --seed`.

- Python, NumPy, and PyTorch RNGs are seeded from one source.
- cuDNN autotuning and TF32 are disabled for reproducible math.
- Dataset augmentations use per-sample deterministic RNG instead of global worker state.
- Training chunk sampling is seeded explicitly, so batch selection is stable across runs.
- Benchmark and analysis sampling paths use stable local seeds as well.

Example:

```bash
python Main/experiments/train_AL_cycle.py --seed 1337
```

Residual limit: exact bitwise identity can still depend on the installed CUDA, cuDNN, PyTorch, and OpenCV versions, plus the GPU architecture. Within the same software and hardware stack, runs should now be reproducible.
