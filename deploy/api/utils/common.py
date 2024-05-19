import os
from functools import lru_cache

import numpy as np
from omegaconf import OmegaConf


def realpath_to_id(path):
    if isinstance(path, int):
        path = str(path)
    return os.path.basename(os.path.splitext(path)[0])


def split_to_equal_chunk(arr: np.array, chunk_size):
    arr = np.array_split(arr, np.ceil(len(arr) / chunk_size))
    return arr


@lru_cache
def read_config(config_path):
    return OmegaConf.load(config_path)
