import typing

import torch

from drools_py.output_strategy.beam_search import BeamSearchOutputStrategyConfig
from drools_py.output_strategy.metropolis_hastings import MetropolisHastingsSamplingConfig
from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.output_strategy_types import OutputStrategyType, OutputStrategyTypeConfigOption
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig
from python_di.configs.autowire import injectable


class ArgmaxSampling(OutputStrategy):
    def sample(self, logits):
        return torch.argmax(logits, dim=-1)


class RandomSampling(OutputStrategy):

    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)

    def sample(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probabilities,
                                 self.output_strategy_config.num_samples.config_option).squeeze()


class TemperatureSampling(OutputStrategy):
    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)
        self.t = output_strategy_config.temperature
        assert self.t > 0

    def sample(self, logits):
        original_shape = logits.shape
        softmax = torch.softmax(logits / self.t, dim=2)
        view = softmax.view(-1, logits.num_kernels(2))
        multinomial = torch.multinomial(view, num_samples=1)
        multinomial_view = multinomial.view(-1)
        return multinomial_view.reshape([original_shape[0], original_shape[1], 1])


class OutputStrategyFactoryConfig(Config):

    def __init__(self,
                 output_strategy_type: OutputStrategyType):
        self.output_strategy_type = output_strategy_type

    @classmethod
    def output_strategy_factory(cls, output_strategy: OutputStrategyTypeConfigOption):
        return OutputStrategyFactoryConfig(
            output_strategy_type=output_strategy.config_option,
        )


class OutputStrategyFactory(ConfigFactory):
    def __init__(self, config_of_item_to_create: OutputStrategyFactoryConfig):
        super().__init__(config_of_item_to_create)
        self.config = config_of_item_to_create
        self.output_strategy_config: typing.Optional[OutputStrategyConfig] = None
        self.beam_search_config: typing.Optional[BeamSearchOutputStrategyConfig] = None
        self.metropolis: typing.Optional[MetropolisHastingsSamplingConfig] = None

    @injectable()
    def output_strategy_configs(self, output_strategy_config: OutputStrategyConfig,
                                beam_search_config: BeamSearchOutputStrategyConfig,
                                metropolis: MetropolisHastingsSamplingConfig):
        self.output_strategy_config = output_strategy_config
        self.beam_search_config = beam_search_config
        self.metropolis = metropolis

    def create(self, **kwargs):
        from drools_py.output_strategy.metropolis_hastings import MetropolisHastingsSampling, \
            MetropolisHastingsSamplingConfig
        from drools_py.output_strategy.sample_rank import SampleAndRank
        from drools_py.output_strategy.stochastic_beam_search import StochasticBeamSearch
        from drools_py.output_strategy.top_k_sampling import TopKSampling
        from drools_py.output_strategy.nucleus_sampling import NucleusSampling

        if self.config.output_strategy_type == OutputStrategyType.Temperature:
            return TemperatureSampling(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.TopK:
            return TopKSampling(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.TopP:
            return NucleusSampling(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.Random:
            return RandomSampling(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.ArgMax:
            return ArgmaxSampling(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.SampleRank:
            return SampleAndRank(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.StochasticBeamSearch:
            return StochasticBeamSearch(self.output_strategy_config)
        elif self.config.output_strategy_type == OutputStrategyType.BeamSearch:
            from drools_py.output_strategy.beam_search import BeamSearch, BeamSearchOutputStrategyConfig
            assert isinstance(self.beam_search_config, BeamSearchOutputStrategyConfig), \
                (f"Specified beam search with incorrect config: {self.beam_search_config.__class__.__name__} "
                 f"config provided.")
            return BeamSearch(self.beam_search_config)
        elif self.config.output_strategy_type == OutputStrategyType.MetropolisHastings:
            assert isinstance(self.metropolis, MetropolisHastingsSamplingConfig), \
                (f"Specified beam search with incorrect config: {self.beam_search_config.__class__.__name__} "
                 f"was provided config.")
            return MetropolisHastingsSampling(self.metropolis)
