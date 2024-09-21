import numpy as np
import torch
from entropy.entropy import spectral_entropy, sample_entropy
from scipy import stats

from drools_py.measure.measure_strategy import DelegatingMeasureStrategy, MeasureStrategyProvider, MeasureStrategy
from python_util.logger.logger import LoggerFacade


def compute_shannon_entropy(time_series, num_bins=50):
    # Compute histogram of the time series
    hist, bin_edges = np.histogram(time_series, bins=num_bins, density=True)

    # Compute the probabilities
    probs = hist / np.sum(hist)

    return stats.entropy(probs)


def compute_spectral_entropy(sf, history, method='welch', normalize=True):
    return spectral_entropy(history, sf, method=method, normalize=normalize)


def compute_sample_entropy(m, history):
    return sample_entropy(history, order=m)


class CrossEntropyMeasureStrategy(DelegatingMeasureStrategy):

    def __init__(self):
        super().__init__(torch.nn.functional.cross_entropy)


class CrossEntropyMeasureStrategyProvider(MeasureStrategyProvider):
    def create_measure_strategy(self):
        return CrossEntropyMeasureStrategy()


class KlDivergenceMeasureStrategy(DelegatingMeasureStrategy):
    def __init__(self):
        super().__init__(torch.nn.functional.kl_div)


class KlDivergenceMeasureStrategyProvider(MeasureStrategyProvider):
    def create_measure_strategy(self):
        return KlDivergenceMeasureStrategy()


class PointwiseKlDivergenceMeasureStrategy(MeasureStrategy):
    def compute_measure(self, scores: torch.Tensor, take_softmax: bool = False, *args, **kwargs):
        """
        Compute the pointwise KL Divergence for attention scores of size [k x d]
        :param take_softmax:
        :param scores:
        :param args:
        :param kwargs:
        :return:
        """
        # Normalize the probabilities to ensure they sum up to 1
        t1 = scores.unsqueeze(1)  # Shape: [k, 1, d]
        t2 = scores.unsqueeze(0)  # Shape: [1, k, d]

        if take_softmax:
            t1 = torch.nn.functional.softmax(t1)
            t2 = torch.nn.functional.softmax(t2)

        # Compute element-wise log and division
        log_t1_over_t2 = torch.log(t1 / t2)

        # Compute KL-Divergence
        kl_div = (t1 * log_t1_over_t2).sum(dim=-1)  # Sum along the last dimension
        return kl_div
