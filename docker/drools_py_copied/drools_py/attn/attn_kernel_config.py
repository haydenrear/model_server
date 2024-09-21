import torch

from codegen.generated_config_models import InDimAttnKernelConfig
from drools_py.attn.attn_kernel import AttnKernelTypesConfigOption
from drools_py.configs.config import Config


class AttnKernelConfig(Config):
    def __init__(self,
                 kernel_type: AttnKernelTypesConfigOption,
                 in_dim:  InDimAttnKernelConfig):
        self.in_dim = in_dim
        self.kernel_type = kernel_type
