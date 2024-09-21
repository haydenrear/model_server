from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class LearningRateSchedulerType(EnumConfigOption):
    StepLR = 0
    OneCycleLR = 1
    ReduceLROnPlateau = 2
    CosineAnnealingWarmRestarts = 3
    CosineWarmup = 4
    SWALR = 5
    CosineAnnealingLR = 6
    NoOp = 7


class LearningRateSchedulerTypeConfigOption(ConfigOption[LearningRateSchedulerType]):
    def __init__(self, config_option: LearningRateSchedulerType):
        super().__init__(config_option)