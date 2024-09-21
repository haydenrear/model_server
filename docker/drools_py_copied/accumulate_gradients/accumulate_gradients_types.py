from drools_py.configs.config_models import ConfigOption, EnumConfigOption
from drools_py.serialize.serializable_types import SerializableEnum


class AccumulateGradientsTypes(EnumConfigOption):
    AccumulateGradients = 0
    DontAccumulateGradients = 1


class AccumulateGradientsConfigOption(ConfigOption[AccumulateGradientsTypes]):
    def __init__(self, config_option: AccumulateGradientsTypes):
        super().__init__(config_option)
