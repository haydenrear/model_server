import numpy as np
from numpy.lib.stride_tricks import as_strided

from drools_py.norm.similarity_fn import cosine


def get_top_k_embeddings(embeddings, centroid,
                         k, similarity=cosine):
    cosine_similarities = similarity(embeddings, centroid)
    top_k_indices = np.argpartition(cosine_similarities, -k)[-k:]
    top = embeddings[top_k_indices]
    return top


def sliding_window(arr, window_size, agg_fn):
    prev_shape = [i for i in arr.shape[-1:]]
    prev_shape.insert(0, window_size)
    prev_shape.insert(0, arr.shape[0] // window_size)
    arr_strided = as_strided(arr, shape=prev_shape)
    return [
        agg_fn(arr) for arr in arr_strided
    ]
