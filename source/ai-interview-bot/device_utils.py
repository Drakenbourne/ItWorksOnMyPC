from functools import lru_cache
from typing import Union

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_pipeline_device() -> Union[int, torch.device]:
    if get_device() == "mps":
        return torch.device("mps")
    return -1
