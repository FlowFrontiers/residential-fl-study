"""Device configuration and synchronized elapsed-time measurements."""

import os
import platform
import random
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build")
    return torch.device(requested)


def configure(config, seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.num_threads)
    threadpool_limits(limits=config.num_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device


def timestamp(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def environment(config):
    import opacus
    import sklearn
    import scipy
    import pandas
    import pyarrow
    device = resolve_device(config.device)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "opacus": str(opacus.__version__),
            "sklearn": str(sklearn.__version__), "cuda": torch.version.cuda,
            "scipy": scipy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
            "device": str(device), "num_threads": config.num_threads,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
