from typing import Optional

import torch.nn.init

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import Kwargs
from drools_py.weight_init.weight_initialization_config_types import WeightInitializationTypes, \
    WeightInitializationTypesConfigOption, WeightInitializationTypesConsumer
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.complex_torch import init_complex_weights


class WeightInitializationConfigArgs(Kwargs):
    pass


class WeightInitializationConfig(Config):

    def __init__(self,
                 initialization_type: WeightInitializationTypesConfigOption,
                 weights_provider: Optional[WeightInitializationTypesConsumer] = None,
                 weight_init_args: Optional[WeightInitializationConfigArgs] = None):
        self.weight_init_args = weight_init_args
        self.weights_provider = weights_provider
        self.initialization_type = initialization_type

    @classmethod
    def weight_initialization_config(cls, weight_types_config: WeightInitializationTypesConfigOption,
                                     weights_provider: Optional[WeightInitializationTypesConsumer] = None,
                                     weight_init_args: Optional[WeightInitializationConfigArgs] = None):
        return WeightInitializationConfig(weight_types_config, weights_provider, weight_init_args)


class WeightInitializationConfigFactory(ConfigFactory):
    def __init__(self, weight_initialization_config: WeightInitializationConfig):
        super().__init__(weight_initialization_config)
        self.weight_initialization_config = weight_initialization_config

    @classmethod
    def weight_initialization_config_factory(cls, weight_types_config: Optional[WeightInitializationTypesConfigOption] = None,
                                             weights_provider: Optional[WeightInitializationTypesConsumer] = None,
                                             weight_init_args: Optional[WeightInitializationConfigArgs] = None):
        return WeightInitializationConfigFactory(
            WeightInitializationConfig.weight_initialization_config(weight_types_config, weights_provider,
                                                                    weight_init_args)
        )

    def create(self, in_module=None, **kwargs):
        init_type = self.weight_initialization_config.initialization_type.config_option

        if init_type == WeightInitializationTypes.XavierNormal:
            return torch.nn.init.xavier_normal_

        elif init_type == WeightInitializationTypes.XavierUniform:
            return torch.nn.init.xavier_uniform_

        elif init_type == WeightInitializationTypes.WeightInitializationConsumer:
            return self.weight_initialization_config.weights_provider

        elif init_type == WeightInitializationTypes.Random:
            return torch.nn.init.normal_

        elif init_type == WeightInitializationTypes.Uniform:
            return torch.nn.init.uniform_

        elif init_type == WeightInitializationTypes.Ones:
            return torch.nn.init.ones_

        elif init_type == WeightInitializationTypes.Zero:
            return torch.nn.init.zeros_

        elif init_type == WeightInitializationTypes.KaimingNormal:
            return torch.nn.init.kaiming_normal_

        elif init_type == WeightInitializationTypes.KaimingUniform:
            return lambda t: torch.nn.init.kaiming_uniform_(t, mode='fan_in', nonlinearity='relu')

        elif init_type == WeightInitializationTypes.Orthogonal:
            return torch.nn.init.orthogonal_

        elif init_type == WeightInitializationTypes.Normal:
            return torch.nn.init.normal_

        elif init_type == WeightInitializationTypes.WeightTying:
            assert self.weight_initialization_config.weights_provider is not None
            return lambda t: self.weight_initialization_config.weights_provider.consume(
                t, **self._get_kwargs(kwargs))

        elif init_type == WeightInitializationTypes.WeightInitializationConsumer:
            return lambda t: self.weight_initialization_config.weights_provider.consume(t, **self._get_kwargs(kwargs))

        elif init_type == WeightInitializationTypes.ComplexFrequencyBands:
            args = self._get_kwargs(kwargs)
            assert 'num_frequency_bands' in args.keys(), \
                f"Kwargs {kwargs} for initialization of complex frequency bands did not include number of bands."
            num_freq_bands = args['num_frequency_bands']
            LoggerFacade.info(f"Found complex weight initialization with {num_freq_bands} number of bands.")
            return lambda t: init_complex_weights(t, num_freq_bands)

        else:
            raise ValueError(f"Unknown weight initialization type: {init_type}")

    def _get_kwargs(self, kwargs):
        if self.weight_initialization_config.weight_init_args is not None:
            for k, v in self.weight_initialization_config.weight_init_args.config_option.items():
                kwargs[k] = v
        return kwargs


class WeightInitialization:

    def __init__(self, weight_initialization_factory: WeightInitializationConfigFactory,
                 **kwargs):
        self.weight_init_args = kwargs
        self.weight_init = weight_initialization_factory.create(**kwargs)

    def initialize(self, size: list[int]) -> torch.Tensor:
        return self.weight_init(torch.zeros(size))
