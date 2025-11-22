import numpy as np
import os

path = "/home/abhiram001/Active_Learning_Multimodal_Image_Matching/Active-Learning-for-Dense-Image-Matching-/datasets/cross_modality/Optical-Infrared/Idx_files/Optical-Infrared_coreset_cycle0.npy"

arr = np.load(path)  # add allow_pickle=True ONLY if it was saved with pickled objects

print(type(arr))
print("shape:", getattr(arr, "shape", None))
print("dtype:", getattr(arr, "dtype", None))

# If it's an array:
if isinstance(arr, np.ndarray):
    print("first 10:", arr[:10])
    print("min/max:", arr.min(), arr.max())
    print("unique (small arrays):", np.unique(arr)[:20])

    # ---- save all index values to CSV ----
    # 1D: keep as is, otherwise flatten
    flat = arr.reshape(-1)

    # output path: same name, .csv instead of .npy
    csv_path = os.path.splitext(path)[0] + ".csv"

    # save as a single column "index"
    np.savetxt(
        csv_path,
        flat,
        fmt="%d",          # change to "%s" if dtype is not integer
        delimiter=",",
        header="index",
        comments=""
    )
    print(f"Saved {flat.shape[0]} indices to {csv_path}")

else:
    raise TypeError("Loaded object is not a numpy.ndarray; can't directly save indices to CSV.")
