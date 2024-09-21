import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class AttnPostProba(EnumConfigOption):
    NoPostProba = enum.auto()
    IFFT = enum.auto()


class AttnPostProbaConfigOption(ConfigOption[AttnPostProba]):

    def __init__(self, config_option: AttnPostProba = AttnPostProba.NoPostProba):
        super().__init__(config_option)


