from drools_py.configs.config_models import EnumConfigOption, ConfigOption
from drools_py.serialize.serializable_types import SerializableEnum


class OutputStrategyType(EnumConfigOption):
    Temperature = 0
    Random = 1
    ArgMax = 2
    BeamSearch = 3
    StochasticBeamSearch = 4
    TopK = 5
    TopP = 6
    SampleRank = 7
    MetropolisHastings = 8


class OutputStrategyTypeConfigOption(ConfigOption[OutputStrategyType]):
    def __init__(self, config_option: OutputStrategyType):
        super().__init__(config_option)