import enum

from drools_py.configs.config_models import EnumConfigOption, ConfigOption


class AttnTypes(EnumConfigOption):
    DistAttn = enum.auto()
    DotProduct = enum.auto()
    DualChunk = enum.auto()
    LinearAttention = enum.auto()
    FlashAttention = enum.auto()
    LocalAttention = enum.auto()
    QuadrangleAttention = enum.auto()
    LshAttention = enum.auto()
    AftAttention = enum.auto()
    VQAttention = enum.auto()
    MsaConv = enum.auto()
    NeighborhoodAttention = enum.auto()
    PerformerFavor = enum.auto()
    HardAttention = enum.auto()
    LongQLora = enum.auto()
    Rwkv = enum.auto()
    # Dual-Guided Spatial-Channel-Temporal
    DGSCT = enum.auto()
    PagedAttention = enum.auto()
    # maxtron
    TrajectoryAttention = enum.auto()
    FourierAttention = enum.auto()
    RingAttention = enum.auto()


class AttnTypesConfigOption(ConfigOption[AttnTypes]):

    def __init__(self, config_option: AttnTypes = AttnTypes.DotProduct):
        super().__init__(config_option)
