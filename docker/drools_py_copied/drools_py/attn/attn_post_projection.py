import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class PostProjectionTypes(EnumConfigOption):
    NoPostProjection = enum.auto()
    FFT = enum.auto()


class AttnPostProjectionConfigOption(ConfigOption[PostProjectionTypes]):

    def __init__(self, config_option: PostProjectionTypes = PostProjectionTypes.NoPostProjection):
        super().__init__(config_option)


