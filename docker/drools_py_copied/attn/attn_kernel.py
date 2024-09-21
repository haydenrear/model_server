import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class KernelTypes(EnumConfigOption):
    NoKernel = enum.auto()
    Gaussian = enum.auto()
    Learned = enum.auto()
    LearnedMoe = enum.auto()
    SelfAttn = enum.auto()


class AttnKernelTypesConfigOption(ConfigOption[KernelTypes]):

    def __init__(self, config_option: KernelTypes = KernelTypes.NoKernel):
        super().__init__(config_option)


