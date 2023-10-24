from typing import Dict, Any

import torch
import numpy as np


# type
LoggedDataset = Dict[str, Any]


def to_tensor(
    arr: np.ndarray,
    dtype: type = float,
    device: str = "cuda:0",
):
    tensor = torch.FloatTensor(arr) if dtype == float else torch.LongTensor(arr)
    return tensor.to(device)


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"
