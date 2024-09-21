import abc

import torch

from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig


class OutputStrategy(abc.ABC):

    def __init__(self, output_strategy_config: OutputStrategyConfig):
        self.output_strategy_config = output_strategy_config

    @abc.abstractmethod
    def sample(self, *args, **kwargs) -> torch.Tensor:
        pass



