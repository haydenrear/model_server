import numbers
from typing import Tuple, Optional

import torch
from drools_py.configs.config import ConfigType

from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory

from codegen.generated_config_models import EmbeddingSizeLayerNorm, RecursiveCatLlmDevice
from drools_py.torch_utils.torch_prov_mod_configs import ExternalTorchModuleConfig
from torch import Tensor
from torch.nn import Parameter, init

from drools_py.configs.config_models import Device, LayerNormEps, LayerNormEnabled, EmbeddingSize
from drools_py.weight_init.weight_initialization_config_types import WeightInitializationTypesLnConfigOption, \
    WeightInitializationTypesLnBiasConfigOption, ComplexWeightInitConsumer, WeightInitializationTypesConfigOption, \
    WeightInitializationTypes
from python_di.configs.prototype import prototype_scope_bean, prototype_factory
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn


def normalize_complex(to_normalize: torch.Tensor):
    abs_vector, angle_vector = torch.abs(to_normalize) + 1, torch.angle(to_normalize)
    norm_abs_vector = abs_vector / torch.max(abs_vector)  # Normalize to maximum radius
    norm_freq_vector = norm_abs_vector * torch.exp(1j * angle_vector)  # Combine normalized radius and original angles
    return norm_freq_vector


def normalize_complex_wavefunction(psi_complex, spatial_variable=None):
    psi_complex = psi_complex.permute(1, 2, 0)
    if spatial_variable is None:
        spatial_variable = torch.linspace(0, 1, psi_complex.size(-1))
        while len(spatial_variable.shape) < len(psi_complex.shape):
            spatial_variable = spatial_variable.unsqueeze(0)
        spatial_variable = spatial_variable.expand(psi_complex.shape)
    else:
        spatial_variable = spatial_variable.permute(1, 2, 0)

    if len(psi_complex.shape) == 2:
        return _do_norm_batch(psi_complex, spatial_variable)

    out = torch.zeros(psi_complex.shape, dtype=torch.complex64)

    for i, s in enumerate(psi_complex):
        out[i] = _do_norm_batch(s, spatial_variable[i])

    return out.permute(2, 0, 1)


def _do_norm_batch(s, spatial_variable):
    squared_magnitude = s ** 2
    norm = torch.sqrt(torch.trapz(squared_magnitude.real, spatial_variable)
                      + torch.trapz(squared_magnitude.imag, spatial_variable) * 1j)
    psi_normalized = s.T / norm
    return psi_normalized.T



@prototype_scope_bean()
class ComplexNormConfig(ExternalTorchModuleConfig):

    @prototype_factory()
    def __init__(self, eps: LayerNormEps, size: EmbeddingSizeLayerNorm,
                 weight_init: ComplexWeightInitConsumer,
                 bias_init: ComplexWeightInitConsumer,
                 do_layer_norm: LayerNormEnabled,
                 device: Device):
        self.device = device.config_option
        self.bias_initialization \
            = WeightInitializationConfigFactory.weight_initialization_config_factory(
            WeightInitializationTypesConfigOption(WeightInitializationTypes.WeightInitializationConsumer),
            bias_init
        )
        self.do_layer_norm = do_layer_norm
        self.weight_initialization \
            = WeightInitializationConfigFactory.weight_initialization_config_factory(
            WeightInitializationTypesConfigOption(WeightInitializationTypes.WeightInitializationConsumer),
            weight_init
        )
        self.eps = eps
        self.size = size

    def to_layer_norm(self):
        ln = ComplexLayerNorm(self)
        ln.bias = self.bias_initialization.create()(ln.bias)
        ln.weight = self.weight_initialization.create()(ln.weight)
        return ln

    @classmethod
    @autowire_fn()
    def layer_norm_config_dim_override(cls,
                                       config_type: ConfigType,
                                       weight_init: ComplexWeightInitConsumer,
                                       bias_init: ComplexWeightInitConsumer,
                                       dim: Optional[EmbeddingSize] = None,
                                       eps: Optional[LayerNormEps] = None,
                                       is_enabled: Optional[LayerNormEnabled] = None):
        return cls.get_build_props(config_type,
                                   size=EmbeddingSizeLayerNorm(dim.config_option) if dim is not None else None,
                                   eps=eps,
                                   do_layer_norm=is_enabled,
                                   weight_init=weight_init,
                                   bias_init=bias_init)


class ComplexLayerNorm(torch.nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, config: ComplexNormConfig) -> None:
        factory_kwargs = {'device': config.device, 'dtype': torch.complex64}
        super().__init__()
        self.layer_norm_eps = config.eps.config_option
        self.weight = Parameter(torch.empty(config.size.config_option, **factory_kwargs))
        self.bias = Parameter(torch.empty(config.size.config_option, **factory_kwargs))

    def forward(self, to_normalize: Tensor) -> Tensor:
        # TODO: try with psi_normalized above also, try doing a group norm with as many groups as there are frequencies
        #       initialized
        abs_vector, angle_vector = torch.abs(to_normalize), torch.angle(to_normalize)
        norm_abs_vector \
            = (abs_vector - torch.mean(abs_vector)) / torch.sqrt((torch.var(abs_vector) + self.layer_norm_eps))
        norm_freq_vector = norm_abs_vector * torch.exp(1j * angle_vector)
        return norm_freq_vector * self.weight + self.bias

