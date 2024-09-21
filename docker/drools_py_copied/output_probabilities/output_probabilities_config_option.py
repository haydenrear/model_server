import enum

from drools_py.configs.config_models import ConfigOption, EnumConfigOption
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope


class OutputProbabilitiesTypes(EnumConfigOption):
    Softmax = enum.auto()
    SoftmaxPlusOne = enum.auto()
    LogSoftmax = enum.auto()
    Sigmoid = enum.auto()
    Complex = enum.auto()


@component(scope=profile_scope)
class OutputProbabilitiesTypeConfigOption(ConfigOption[OutputProbabilitiesTypes]):
    def __init__(self, config_option: OutputProbabilitiesTypes = OutputProbabilitiesTypes.Softmax):
        super().__init__(config_option)
