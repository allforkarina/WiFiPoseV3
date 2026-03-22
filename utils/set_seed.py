"""Utility: set random seeds for reproducibility.

Usage:
    from utils.set_seed import set_seed
    set_seed(42, deterministic=True)
"""
import os
import random
import numpy as np

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for python, numpy and (optionally) torch to improve reproducibility.

    Args:
        seed: integer seed
        deterministic: if True, try to enforce deterministic behavior for CUDA/cuDNN (if PyTorch present)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            # default behavior: allow cudnn benchmark for performance
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except Exception:
        # torch may not be installed in minimal environments; ignore if absent
        pass

    print(f"[set_seed] seed={seed}, deterministic={deterministic}")

__all__ = ["set_seed"]
