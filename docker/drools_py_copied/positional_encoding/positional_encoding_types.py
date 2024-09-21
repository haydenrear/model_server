from enum import auto

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class PositionalEncodingTypes(EnumConfigOption):
    Cosine = auto()
    Complex = auto()
    Rotary = auto()
    XPos = auto()
    MistralRope = auto()
    Alibi = auto()


class PositionalEncodingTypesConfigOption(ConfigOption[PositionalEncodingTypes]):
    def __init__(self, config_option: PositionalEncodingTypes = PositionalEncodingTypes.Cosine):
        super().__init__(config_option)
