import abc
from typing import Optional

import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA

from drools_py.measure.measure_strategy import MeasureStrategyProvider, DelegatingMeasureStrategy, MeasureStrategy
from drools_py.configs.config import Config
from python_util.numpy_util.indexing import index_last_dimension_iterator


def normalize_vectors(vectors):
    # Function to normalize a batch of vectors
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def cosine_centroid(cluster):
    # Normalize all vectors in the cluster
    normalized_vectors = normalize_vectors(cluster)

    # Sum all vectors
    centroid = np.sum(normalized_vectors, axis=0)

    # Normalize centroid
    centroid = centroid / np.linalg.norm(centroid)

    return centroid


def compute_mean_centroid(cluster):
    return np.mean(cluster, axis=0)


def compute_median_centroid(cluster):
    return np.median(cluster, axis=0)


def compute_medoid_centroid(cluster):
    pairwise_distances = np.linalg.norm(cluster[:, None] - cluster, axis=-1)
    return cluster[np.argmin(pairwise_distances.sum(axis=0))]


def compute_eigencentroid(cluster):
    pca = PCA(n_components=1)
    reduced = pca.fit_transform(cluster)
    index_closest_to_mean = np.argmin(np.abs(reduced - np.mean(reduced)))
    return cluster[index_closest_to_mean]


def compute_spectral_centroid(cluster):
    pass


class CentroidStrategyProvider(MeasureStrategyProvider, abc.ABC):
    pass


class CentroidStrategyConfig(Config):
    def __init__(self,
                 per_dimension_centroid: bool,
                 centroid_strategy_provider: CentroidStrategyProvider,
                 per_dimension_centroid_strategy: dict[range, CentroidStrategyProvider] = None):
        """
        To produce the center of the cluster, you calculate the centroid using some metric.
        :param per_dimension_centroid: Whether to calculate the centroid seperately for each dimension.
        :param centroid_strategy_provider: The provider of a function that computes the centroid metric.
        :param per_dimension_centroid_strategy: If you ever want to compute the centroid in a different metric for each
        dimension. For example if you concatted some features.
        """
        self.centroid_strategy_provider = centroid_strategy_provider
        self.per_dimension_centroid_strategy = per_dimension_centroid_strategy
        self.per_dimension_centroid = per_dimension_centroid


class CentroidStrategy(MeasureStrategy):
    def __init__(self, centroid_strategy_config: CentroidStrategyConfig, *args, **kwargs):
        super().__init__(centroid_strategy_config)
        self.measure_strategy_config = centroid_strategy_config
        if centroid_strategy_config.per_dimension_centroid_strategy:
            self.per_dimension: dict[int, MeasureStrategy] = {
                k: v.create_measure_strategy(*args, **kwargs) for k, v
                in centroid_strategy_config.per_dimension_centroid_strategy.items()
            }
            self.delegate: Optional[MeasureStrategy] = None
        else:
            self.delegate: Optional[MeasureStrategy] = (centroid_strategy_config.centroid_strategy_provider
                             .create_measure_strategy(*args, **kwargs))
            self.per_dimension = None
        self.centroid_config = centroid_strategy_config

    def compute_measure(self, cluster: np.array, *args, **kwargs):
        out = np.empty(cluster.shape[-(len(cluster.shape) - 1):])
        if self.measure_strategy_config.per_dimension_centroid:
            for i in index_last_dimension_iterator(cluster):
                to_compute_centroid = cluster[i]
                delegator = self.per_dimension[i].compute_measure(to_compute_centroid, *args, **kwargs)
                index = i[-(len(cluster.shape) - 1):]
                out_index = tuple(index)
                out[out_index] = delegator
            return out
        else:
            assert self.delegate is not None
            return self.delegate.compute_measure(cluster, *args, **kwargs)


class MeanCentroidStrategyProvider(CentroidStrategyProvider):

    def create_measure_strategy(self, config):
        return CentroidStrategy(config, compute_mean_centroid)

