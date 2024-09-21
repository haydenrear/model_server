import torch.nn.init

from codegen.generated_config_models import OutputProbabilitiesTypeEdgeClassifier
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import Dim
from drools_py.output_probabilities.output_probabilities_config_option import OutputProbabilitiesTypes, \
    OutputProbabilitiesTypeConfigOption


def softmax_plus_one(x: torch.Tensor, dim: int):
    e_x = torch.exp(x)
    return e_x / (1 + e_x.sum(dim=dim, keepdim=True))


def complex_softmax(x: torch.Tensor, dim: int):
    conj_ = (x * x.conj()) ** 2
    return conj_ / (torch.sum(conj_, dim=dim, keepdim=True))


class OutputProbabilitiesConfig(Config):

    def __init__(self, output_probabilities: OutputProbabilitiesTypeConfigOption,
                 dim_for_prob: Dim = 2):
        self.dim_for_prob = dim_for_prob
        self.output_probabilities = output_probabilities

    @classmethod
    def output_probabilities_config(cls):
        return OutputProbabilitiesConfig(
            output_probabilities=OutputProbabilitiesTypeConfigOption(OutputProbabilitiesTypes.Sigmoid)
        )


class OutputProbabilitiesConfigFactory(ConfigFactory):
    def __init__(self, output_probabilities_config: OutputProbabilitiesConfig):
        super().__init__(output_probabilities_config)
        self.output_probabilities_config = output_probabilities_config

    def create(self, dim: int = None, **kwargs):
        assert dim is not None or self.output_probabilities_config.dim_for_prob is not None, \
            "Output features not provided."
        if dim is None:
            dim = self.output_probabilities_config.dim_for_prob
        if self.output_probabilities_config.output_probabilities.config_option == OutputProbabilitiesTypes.Softmax:
            return torch.nn.functional.softmax if dim is None else torch.nn.Softmax(dim=dim)
        elif self.output_probabilities_config.output_probabilities.config_option == OutputProbabilitiesTypes.LogSoftmax:
            return torch.nn.functional.log_softmax if dim is None else torch.nn.LogSoftmax(dim=dim)
        elif (self.output_probabilities_config.output_probabilities.config_option
              == OutputProbabilitiesTypes.SoftmaxPlusOne):
            return softmax_plus_one
        elif (self.output_probabilities_config.output_probabilities.config_option
              == OutputProbabilitiesTypes.Sigmoid):
            return torch.nn.Sigmoid() if dim is not None else torch.nn.functional.sigmoid
        elif (self.output_probabilities_config.output_probabilities.config_option
              == OutputProbabilitiesTypes.Complex):
            return complex_softmax

    @classmethod
    def output_probabilities_config_factory(cls, output_probabilities_config: OutputProbabilitiesTypeConfigOption):
        return OutputProbabilitiesConfigFactory(OutputProbabilitiesConfig(output_probabilities_config))


class OutputProbabilities:

    def __init__(self, output_probabilities_factory: OutputProbabilitiesConfigFactory,
                 **kwargs):
        self.output_probabilities = output_probabilities_factory.create(**kwargs)

    def initialize(self, size: list[int]) -> torch.Tensor:
        return self.output_probabilities(torch.zeros(size))
