import abc

from drools_py.configs.config import ConfigType

from drools_py.attn.attn_kernel import KernelTypes
from drools_py.attn.attn_kernel_config import AttnKernelConfig
from drools_py.configs.config_factory import ConfigFactory
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_util.logger.logger import LoggerFacade


class AttnKernelFunctionConfigFactory(ConfigFactory, abc.ABC):

    def __init__(self, attn_kernel_config: AttnKernelConfig):
        super().__init__(attn_kernel_config)
        self.attn_kernel_config = attn_kernel_config
        self.model = None

    @classmethod
    @autowire_fn()
    def attn_kernel_fn_factory(cls, config_type: ConfigType, config: AttnKernelConfig):
        return AttnKernelFunctionConfigFactory(config)

    def create(self, **kwargs):
        if self.attn_kernel_config.kernel_type == KernelTypes.LearnedMoe:
            from drools_py.attn.moe_attn_kernel import MixtureOfExpertsScoreKernel, MixtureOfExpertsScoreKernelConfig
            assert isinstance(self.attn_kernel_config, MixtureOfExpertsScoreKernelConfig)
            LoggerFacade.debug("Creating MOE kernel.")
            self.model = MixtureOfExpertsScoreKernel(self.attn_kernel_config)
        elif self.attn_kernel_config.kernel_type == KernelTypes.SelfAttn:
            from drools_py.attn.self_attn_fourier_kernel import SelfAttnFourierKernelConfig, SelfAttnFourierKernel
            assert isinstance(self.attn_kernel_config, SelfAttnFourierKernelConfig)
            LoggerFacade.debug("Creating fourier self-attn kernel.")
            self.model = SelfAttnFourierKernel(self.attn_kernel_config)

        return self.model
