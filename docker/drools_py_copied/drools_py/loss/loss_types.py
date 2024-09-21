import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption
from drools_py.serialize.serializable_types import SerializableEnum


class LossTypes(EnumConfigOption):
    CrossEntropy = enum.auto()
    KlDivergence = enum.auto()
    LlmCrossEntropyWithPenalty = enum.auto()


class LossTypesConfigOption(ConfigOption[LossTypes]):
    def __init__(self, config_option: LossTypes):
        super().__init__(config_option)
