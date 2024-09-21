import abc

import numpy as np
import torch

from drools_py.measure.measure_strategy import DelegatingMeasureStrategy, MeasureStrategyProvider, compute_mean_measure

N = int


class IncrementalMeasureStrategyData:
    def __init__(self,
                 num_used: N,
                 prev: np.array):
        self.prev = prev
        self.num_used = num_used


class IncrementalMeasureStrategy(DelegatingMeasureStrategy, abc.ABC):

    @abc.abstractmethod
    def update_measurement(self, new: np.array,
                           metadata: IncrementalMeasureStrategyData):
        pass


class IncrementalMeanMeasureStrategy(IncrementalMeasureStrategy):
    def update_measurement(self, new: np.array,
                           metadata: IncrementalMeasureStrategyData):
        new_sum = np.sum(new, axis=0)
        return (metadata.prev * metadata.num_used + new_sum) / (metadata.num_used + new)


class MeanMeasureStrategyProvider(MeasureStrategyProvider):

    def create_measure_strategy(self, config):
        return IncrementalMeanMeasureStrategy(config, compute_mean_measure)

