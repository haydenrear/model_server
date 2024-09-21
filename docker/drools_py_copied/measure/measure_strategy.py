import abc

import numpy as np

from drools_py.configs.config import Config


def compute_mean_measure(input: np.array) -> np.array:
    return np.mean(input, axis=0)


class MeasureStrategyProvider(abc.ABC):
    @abc.abstractmethod
    def create_measure_strategy(self, *args, **kwargs):
        pass


class MeasureStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_measure(self, *args, **kwargs):
        pass


class DelegatingMeasureStrategy(MeasureStrategy):

    def __init__(self, measure_strategy_delegator, *args, **kwargs):
        self.measure_strategy_delegator = measure_strategy_delegator
        self.args = args
        self.kwargs = kwargs

    def compute_measure(self, *args, **kwargs):
        return self.measure_strategy_delegator(*self.args, *args, **self.kwargs, **kwargs)


class MeasureStrategyProvidedMeasureStrategy(DelegatingMeasureStrategy):

    def __init__(self, measure_strategy_provider: MeasureStrategyProvider,
                 *args, **kwargs):
        self.measure_strategy_delegator = measure_strategy_provider.create_measure_strategy(*args, **kwargs)
        super().__init__(self.measure_strategy_delegator)

