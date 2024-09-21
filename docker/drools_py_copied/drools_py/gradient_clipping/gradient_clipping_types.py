import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class GradientClippingTypes(EnumConfigOption):
    Value = enum.auto()
    SkipGradientClipping = enum.auto()


class GradientClippingConfigOption(ConfigOption[GradientClippingTypes]):
    def __init__(self, config_option: GradientClippingTypes):
        super().__init__(config_option)
