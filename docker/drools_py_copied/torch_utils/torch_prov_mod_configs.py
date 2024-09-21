import abc
import dataclasses
from typing import Optional

import torch.nn
from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory

from codegen.generated_config_models import EmbeddingSizeLayerNorm
from drools_py.configs.config_models import EdgeDim, NumAttnHeads, OutDim, DoSkipConnections, LayerNormEps, InDim, \
    EmbeddingSize, LayerNormEnabled
from drools_py.configs.config import Config, ConfigType
from drools_py.serialize.serializer import SerializerType
from drools_py.weight_init.weight_initialization_config_types import WeightInitializationTypesLnConfigOption, \
    WeightInitializationTypesLnBiasConfigOption, WeightInitializationTypes
from python_di.configs.prototype import prototype_scope_bean, prototype_factory
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn


class ExternalTorchModuleConfig(Config, abc.ABC):
    pass


class MetaConfig(Config):

    def __init__(self, class_type: str):
        self.class_type = class_type

    def to_dict(self):
        dict_created = {
            'class_type': self.class_type,
            "value_type": [SerializerType.Reflecting.name],
            "clzz": MetaConfig.__name__,
            "module": MetaConfig.__module__
        }
        return dict_created

    def to_self_dictionary(self) -> dict:
        return self.to_dict()


@prototype_scope_bean()
class LayerNormConfig(ExternalTorchModuleConfig):

    @prototype_factory()
    def __init__(self, eps: LayerNormEps, size: EmbeddingSizeLayerNorm,
                 weight_init: WeightInitializationTypesLnConfigOption,
                 bias_init: WeightInitializationTypesLnBiasConfigOption,
                 do_layer_norm: LayerNormEnabled):
        if bias_init is None:
            bias_init = WeightInitializationTypesLnBiasConfigOption(WeightInitializationTypes.Zero)
        if weight_init is None:
            weight_init = WeightInitializationTypesLnConfigOption(WeightInitializationTypes.Ones)
        self.bias_initialization \
            = WeightInitializationConfigFactory.weight_initialization_config_factory(bias_init)
        self.do_layer_norm = do_layer_norm
        self.weight_initialization \
            = WeightInitializationConfigFactory.weight_initialization_config_factory(weight_init)
        self.eps = eps
        self.size = size

    def to_layer_norm(self) -> torch.nn.LayerNorm:
        ln = torch.nn.LayerNorm(self.size.config_option,
                                self.eps.config_option)
        ln.bias = self.bias_initialization.create()(ln.bias)
        ln.weight = self.weight_initialization.create()(ln.weight)
        return ln

    @classmethod
    @autowire_fn()
    def layer_norm_config_dim_override(cls,
                                       config_type: ConfigType,
                                       weight_init: WeightInitializationTypesLnConfigOption,
                                       bias_init: WeightInitializationTypesLnBiasConfigOption,
                                       dim: Optional[EmbeddingSize] = None,
                                       eps: Optional[LayerNormEps] = None,
                                       is_enabled: Optional[LayerNormEnabled] = None):
        return cls.build_config_type_config(config_type,
                                            size=EmbeddingSizeLayerNorm(dim.config_option) if dim is not None else None,
                                            eps=eps,
                                            do_layer_norm=is_enabled,
                                            weight_init=weight_init,
                                            bias_init=bias_init)


class MultiHeadAttentionModuleConfig(ExternalTorchModuleConfig):

    def __init__(self,
                 embed_dim: EdgeDim,
                 kdim: EdgeDim,
                 vdim: EdgeDim,
                 n_attn_heads: NumAttnHeads):
        self.n_attn_heads = n_attn_heads
        self.vdim = vdim
        self.kdim = kdim
        self.embed_dim = embed_dim

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class LinearTorchModuleConfig(ExternalTorchModuleConfig):

    def __init__(self,
                 in_features: Optional[EdgeDim] = None,
                 out_features: Optional[OutDim] = None,
                 bias: Optional[DoSkipConnections] = None):
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ProbabilityLayerTchModuleConfig(ExternalTorchModuleConfig):
    """
    Softmax
    """

    def __init__(self,
                 in_features: Optional[EdgeDim] = None):
        self.in_features = in_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class SequentialModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: list[Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ModuleListModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: list[Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ModuleDictModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: dict[str, Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ActivationFunctionModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, in_features: EdgeDim):
        self.in_features = in_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass
