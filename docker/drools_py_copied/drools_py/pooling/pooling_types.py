import enum

from drools_py.configs.config_models import ConfigOption, EnumConfigOption


class PoolType(EnumConfigOption):
    Mean = enum.auto()
    Sum = enum.auto()
    Max = enum.auto()
    NoPool = enum.auto()


class PoolTypeConfigOption(ConfigOption[PoolType]):
    def __init__(self, config_option: PoolType = PoolType.Mean):
        super().__init__(config_option)
