from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class ActivationTypes(EnumConfigOption):
    ReLU = 0
    LeakyReLU = 1
    Sigmoid = 2
    TanH = 3
    PReLU = 4
    ELU = 5
    GELU = 6
    Softmax = 7


class ActivationTypeConfigOption(ConfigOption[ActivationTypes]):
    def __init__(self, config_option: ActivationTypes):
        super().__init__(config_option)
