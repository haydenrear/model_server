from drools_py.configs.config_models import EnumConfigOption, ConfigOption
from drools_py.serialize.serializable_types import SerializableEnum


class OptimizerType(EnumConfigOption):
    Adam = 0
    AdamW = 1
    RAdam = 2
    Adan = 3


class OptimizerTypeConfigOption(ConfigOption[OptimizerType]):
    def __init__(self, config_option: OptimizerType):
        super().__init__(config_option)
