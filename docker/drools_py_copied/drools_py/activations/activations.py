import torch.nn.init

from drools_py.activations.activation_types import ActivationTypes
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory


class ActivationConfig(Config):

    def __init__(self,
                 initialization_type: ActivationTypes):
        self.initialization_type = initialization_type

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return ActivationConfig.update_override(
            ActivationConfig(ActivationTypes.ReLU).to_self_dictionary(),
            kwargs
        )


class ActivationConfigFactory(ConfigFactory):
    def __init__(self, activation_config: ActivationConfig):
        super().__init__(activation_config)
        self.activation_config = activation_config

    def create(self, **kwargs):
        init_type = self.activation_config.initialization_type

        if init_type == ActivationTypes.ReLU:
            return torch.nn.ReLU()

        elif init_type == ActivationTypes.Softmax:
            return torch.nn.Softmax()

        elif init_type == ActivationTypes.GELU:
            return torch.nn.GELU()

        elif init_type == ActivationTypes.TanH:
            return torch.nn.Tanh()

        elif init_type == ActivationTypes.ELU:
            return torch.nn.ELU()

        elif init_type == ActivationTypes.PReLU:
            return torch.nn.PReLU()

        elif init_type == ActivationTypes.Sigmoid:
            return torch.nn.Sigmoid()
        elif init_type == ActivationTypes.LeakyReLU:
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown weight initialization type: {init_type}")

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return ActivationConfigFactory.update_override(ActivationConfigFactory(
            ActivationConfig.build_test_config()
        ).to_self_dictionary(), kwargs)


class ActivationDelegate:

    def __init__(self, activation_config_factory: ActivationConfigFactory,
                 **kwargs):
        self.activation_fn_args = kwargs
        self.activation_fn = activation_config_factory.create(**kwargs)

    def initialize(self, input_val: torch.Tensor) -> torch.Tensor:
        return self.activation_fn(input_val)
