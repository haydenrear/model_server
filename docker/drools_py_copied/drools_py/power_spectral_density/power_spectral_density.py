import numpy as np
from scipy import signal

from drools_py.measure.measure_strategy import DelegatingMeasureStrategy, MeasureStrategyConfig, MeasureStrategyProvider
from drools_py.configs.config import Config


def compute_power_spectral_density(history):
    frequencies, psd_values = signal.welch(history, noverlap=99, axis=0)
    return frequencies, psd_values


class PowerSpectralDensityConfig(MeasureStrategyConfig):
    def __init__(self,
                 measure_strategy_provider: MeasureStrategyProvider,
                 num_overlap: int):
        super().__init__(measure_strategy_provider)
        self.num_overlap = num_overlap


class PowerSpectralDensityMeasureStrategyProvider(MeasureStrategyProvider):

    def __init__(self,
                 measure_strategy_config: PowerSpectralDensityConfig):
        self.measure_strategy_config = measure_strategy_config

    def create_measure_strategy(self, config):
        return PowerSpectralDensityMeasureStrategy(config)


class PowerSpectralDensityMeasureStrategy(DelegatingMeasureStrategy):

    def __init__(self, measure_strategy_config: PowerSpectralDensityConfig):
        super().__init__(measure_strategy_config, compute_power_spectral_density)
        self.args = {
            "num_overlap": measure_strategy_config.num_overlap
        }

    def compute_measure(self, cluster: np.array, **kwargs):
        self.measure_strategy_delegator(cluster, **Config.update_override(kwargs, self.args))

