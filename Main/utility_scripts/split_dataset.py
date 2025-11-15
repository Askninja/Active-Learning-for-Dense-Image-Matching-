import numpy as np
import os.path as osp

seed = 42
data_root = '/home/abhiram001/active_learning/abhiram/AMD_ab/datasets/cross_modality/Optical-Optical'

num_sample = 200
counts = (150, 20, 20, 10)

rng = np.random.default_rng(seed)
total = rng.permutation(np.arange(1, num_sample + 1))

train = total[:counts[0]]
test = total[counts[0]:counts[0]+counts[1]]
val = total[counts[0]+counts[1]:counts[0]+counts[1]+counts[2]]
preseed = total[counts[0]+counts[1]+counts[2]:counts[0]+counts[1]+counts[2]+counts[3]]

np.save(osp.join(data_root, 'train_idx.npy'), train)
np.save(osp.join(data_root, 'test_idx.npy'), test)
np.save(osp.join(data_root, 'val_idx.npy'), val)
np.save(osp.join(data_root, 'preseed_idx.npy'), preseed)
print(train.size, test.size, val.size, preseed.size)
