import numpy as np


def index_last_dimension_iterator(input_arr: np.array):
    next_slice = []
    for i in range(input_arr.shape[-1]):

        shape_ = input_arr.shape[:len(input_arr.shape)-1]
        next_slice.extend([range(0, shape_item) for shape_item in shape_])
        next_slice.append(i)
        yield tuple(next_slice)
        next_slice.clear()