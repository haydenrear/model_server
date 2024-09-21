import abc
import typing
from typing import Optional

import injector
import torch.nn

import python_util.reflection.reflection_utils
from codegen.generated_config_models import EmbeddingSizeLayerNorm
from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_models import EdgeDim, OutDim, DoSkipConnections, NumAttnHeads, LayerNormEps
from python_di.inject.context_builder.inject_ctx import inject_context_di
from python_di.inject.injector_provider import InjectionContextInjector
from python_util.logger.logger import LoggerFacade


class TchProvModuleConfigAdapter(abc.ABC):
    @abc.abstractmethod
    def create_config(self, module: torch.nn.Module) -> Optional[Config]:
        pass

    @abc.abstractmethod
    def supports(self, module: torch.nn.Module) -> bool:
        pass


class LinearLayerConfigAdapterProv(TchProvModuleConfigAdapter):

    def create_config(self, module: torch.nn.Module) -> Optional[Config]:
        module: torch.nn.Linear = module

        from drools_py.torch_utils.torch_prov_mod_configs import LinearTorchModuleConfig
        return LinearTorchModuleConfig(EdgeDim(module.in_features),
                                       OutDim(module.out_features),
                                       bias=DoSkipConnections(module.bias is not None))

    def supports(self, module: torch.nn.Module) -> bool:
        return isinstance(module, torch.nn.Linear)


class LayerNormConfigAdapterProv(TchProvModuleConfigAdapter):

    def supports(self, module: torch.nn.Module) -> bool:
        return isinstance(module, torch.nn.LayerNorm)

    def create_config(self, module: torch.nn.Module) -> Optional[Config]:
        module: torch.nn.LayerNorm = module
        from drools_py.torch_utils.torch_prov_mod_configs import LayerNormConfig
        return LayerNormConfig(LayerNormEps(module.eps),
                               EmbeddingSizeLayerNorm(module.normalized_shape))


class MultiHeadAttentionConfigAdapterProv(TchProvModuleConfigAdapter):

    def supports(self, module: torch.nn.Module) -> bool:
        return isinstance(module, torch.nn.MultiheadAttention)

    def create_config(self, module: torch.nn.Module) -> Optional[Config]:
        module: torch.nn.MultiheadAttention = module
        kdim = module.kdim if module.kdim else module.embed_dim
        vdim = module.vdim if module.vdim else module.embed_dim
        from drools_py.torch_utils.torch_prov_mod_configs import MultiHeadAttentionModuleConfig
        return MultiHeadAttentionModuleConfig(EdgeDim(module.embed_dim),
                                              EdgeDim(kdim),
                                              EdgeDim(vdim),
                                              NumAttnHeads(module.num_heads))


class TchConfigAdapterProvider(abc.ABC):
    @property
    @abc.abstractmethod
    def adapters(self) -> list[TchProvModuleConfigAdapter]:
        pass

    @adapters.setter
    @abc.abstractmethod
    def adapters(self, adapters: list[TchProvModuleConfigAdapter]):
        pass


class MetaConfigIdTchConfigAdapter(TchProvModuleConfigAdapter):

    def create_config(self, module: torch.nn.Module) -> Optional[Config]:
        from drools_py.torch_utils.torch_prov_mod_configs import MetaConfig
        return MetaConfig(module.__class__.__name__)

    def supports(self, module: torch.nn.Module) -> bool:
        return True


class TchConfigAdapters(TchConfigAdapterProvider):

    @injector.inject
    def __init__(self, adapters: typing.List[TchProvModuleConfigAdapter]):
        self._adapters = adapters

    @property
    def adapters(self) -> list[TchProvModuleConfigAdapter]:
        return self._adapters

    @adapters.setter
    def adapters(self, adapters: list[TchProvModuleConfigAdapter]):
        self._adapters = adapters


@inject_context_di()
def create_external_torch_module_config_adapter(module: torch.nn.Module,
                                                config_type: ConfigType,
                                                ctx: typing.Optional[InjectionContextInjector] = None) -> Optional[Config]:
    adapters = ctx.get_interface(TchConfigAdapterProvider)
    for v in filter(
        lambda value: value is not None,
        python_util.reflection.reflection_utils.get_all_fn_param_types_no_default(module.__init__).values()
    ):
        if hasattr(v, '__bases__'):
            if Config in v.__bases__ or v == Config or v in v.__mro__:
                try:
                    LoggerFacade.info(f"Creating config for {v} with config type {config_type} and module "
                                      f"{module.__class__.__name__}.")
                    built_config = v.build_config_type_config(config_type)
                    if built_config is not None:
                        return built_config
                except Exception as e:
                    LoggerFacade.error(f"Failed to build config type for {v} and {config_type} with {e}.")
    for module_adapter in adapters.adapters:
        if module_adapter.supports(module):
            try:
                created = module_adapter.create_config(module)
                if created is not None:
                    return created
            except Exception as e:
                LoggerFacade.error(f"Failed to build config type for {module.__class__.__name__} and {config_type} "
                                   f"with {e}.")

    from drools_py.torch_utils.torch_prov_mod_configs import MetaConfig
    return MetaConfig(module.__class__.__name__)

