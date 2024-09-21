import abc

import torch

from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig


class SampleAndRank(OutputStrategy, abc.ABC):
    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)
        self.num_samples = output_strategy_config.num_samples

    @abc.abstractmethod
    def ranker(self, samples, original_logits):
        """
        This function should return a score for each sample
        It could use another neural network, heuristic, etc.
        :param samples:
        :param original_logits:
        :return:
        """
        pass

    def sample(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        samples = torch.multinomial(probabilities, self.num_samples, replacement=True)
        scores = self.ranker(samples, logits)
        top_samples = samples.gather(1, torch.argmax(scores, dim=1).unsqueeze(1))
        return top_samples.squeeze()
