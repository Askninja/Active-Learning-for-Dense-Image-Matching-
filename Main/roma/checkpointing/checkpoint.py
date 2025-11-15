import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger
import gc

import roma

class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, suffix):
        return os.path.join(self.dir, f"{self.name}_{suffix}.pth")

    def save(self, model, optimizer, lr_scheduler, n):
        if roma.RANK == 0:
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model = model.module
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(states, self._path("latest"))
            logger.info(f"Saved states {list(states.keys())}, at step {n}")

    def save_best(self, model, optimizer, lr_scheduler, n):
        if roma.RANK == 0:
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model = model.module
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(states, self._path("best"))
            logger.info(f"Saved states {list(states.keys())}, at step {n}")

    def load(self, model, optimizer, lr_scheduler, n):
        latest = self._path("latest")
        if os.path.exists(latest) and roma.RANK == 0:
            states = torch.load(latest, map_location="cpu")
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, n
