import numpy as np


def from_dtype_name(name: str) -> np.dtype:
    if name.lower() == 'float32':
        return np.float32
    if name.lower() == 'int64':
        return np.int64
    return np.float32