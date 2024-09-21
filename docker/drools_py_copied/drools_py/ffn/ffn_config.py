import typing

import torch

from codegen.generated_config_models import DropoutFfn, LayerNormFirstFfn, InDimFfn, EmbeddingSizeFfn, \
    OutDimFfn, \
    InDimLinearLayer, OutDimLinearLayer, FfnIncludeBias, FfnActivationTypes, InDimDense, NumLayersDenseFfn, OutDimDense, \
    FfnDType, DenseDType, LinearLayerDType, LinearLayerWeightInitializationLn, LinearLayerWeightInitializationLnBias, \
    LinearLayerWeightInitialization, LinearLayerBiasInitialization, LinearLayerModuleEnabled
from drools_py.activations.activations import ActivationConfigFactory, ActivationConfig
from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_models import ModuleEnabled
from drools_py.torch_utils.torch_prov_mod_configs import LayerNormConfig
from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory
from python_di.configs.prototype import prototype_scope_bean, prototype_factory
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_di.inject.profile_composite_injector.scopes.prototype_scope import prototype_scope_decorator_factory


@prototype_scope_bean()
class LinearLayerConfig(Config):

    @prototype_factory()
    def __init__(self,
                 in_features: InDimLinearLayer,
                 bias: FfnIncludeBias,
                 out_dim: OutDimLinearLayer,
                 linear_layer_dtype: LinearLayerDType,
                 weight_initialization: WeightInitializationConfigFactory,
                 bias_initialization: WeightInitializationConfigFactory,
                 do_module: ModuleEnabled):
        self.bias_initialization = bias_initialization
        self.do_module = do_module
        self.weight_initialization = weight_initialization
        self.linear_layer_dtype = linear_layer_dtype
        self.bias = bias
        self.out_dim = out_dim
        self.in_features = in_features

    @classmethod
    @autowire_fn()
    def linear_layer_config(cls, config_type: ConfigType,
                            input_dim: InDimLinearLayer,
                            out_dim: OutDimLinearLayer,
                            include_bias: FfnIncludeBias,
                            torch_dtype: LinearLayerDType,
                            weight_initialization: LinearLayerWeightInitialization,
                            bias_initialization: LinearLayerBiasInitialization,
                            do_module: LinearLayerModuleEnabled):
        return LinearLayerConfig(
            in_features=LinearLayerConfig.get_input_dim(input_dim),
            bias=include_bias,
            out_dim=cls.get_out_dim(input_dim, out_dim),
            linear_layer_dtype=torch_dtype,
            weight_initialization=WeightInitializationConfigFactory.weight_initialization_config_factory(weight_initialization),
            bias_initialization=WeightInitializationConfigFactory.weight_initialization_config_factory(bias_initialization),
            do_module=do_module
        )

    @classmethod
    def get_out_dim(cls, input_dim, out_dim):
        if out_dim is None:
            out_dim = OutDimLinearLayer(
                input_dim.config_option if isinstance(input_dim, InDimLinearLayer) else input_dim)
        elif not isinstance(out_dim, OutDimLinearLayer):
            return OutDimLinearLayer(out_dim)
        return out_dim

    @classmethod
    def get_input_dim(cls, input_dim):
        return input_dim if isinstance(input_dim, InDimLinearLayer) else InDimLinearLayer(input_dim)

    def to_linear_layer(self) -> typing.Union[torch.nn.Linear, typing.Callable]:
        if self.do_module is None or not self.do_module.config_option:
            return lambda x: x
        else:
            return torch.nn.Linear(in_features=self.in_features.config_option,
                                   out_features=self.out_dim.config_option,
                                   bias=self.bias.config_option,
                                   dtype=self.linear_layer_dtype.config_option)


@prototype_scope_bean()
class FfnConfig(Config):

    @prototype_factory()
    def __init__(self,
                 dropout_1: DropoutFfn,
                 dropout_2: DropoutFfn,
                 activation: FfnActivationTypes,
                 input_dim: InDimFfn,
                 hidden_dim: EmbeddingSizeFfn,
                 output_dim: OutDimFfn,
                 dtype: FfnDType,
                 layer_norm_first: LayerNormFirstFfn,
                 layer_norm: LayerNormConfig,
                 linear_config_1: LinearLayerConfig,
                 linear_config_2: LinearLayerConfig):
        self.linear_config_2 = linear_config_2
        self.linear_config_1 = linear_config_1
        self.layer_norm_first = layer_norm_first
        self.layer_norm = layer_norm
        self.dtype = dtype
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.activation = ActivationConfigFactory(ActivationConfig(activation.config_option)) \
            if not isinstance(activation, ActivationConfigFactory) \
            else activation
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @classmethod
    @autowire_fn()
    def ffn_with_override(cls,
                          config_type: ConfigType,
                          dim: EmbeddingSizeFfn,
                          layer_norm_first: LayerNormFirstFfn,
                          in_dim: InDimFfn,
                          dtype: FfnDType,
                          layer_norm_config: LayerNormConfig,
                          linear_1_config: LinearLayerConfig,
                          linear_2_config: LinearLayerConfig,
                          ffn_activation_types: FfnActivationTypes):
        if in_dim is None:
            in_dim = dim
        return FfnConfig.build_config_type_config(
            config_type,
            input_dim=in_dim,
            output_dim=OutDimFfn(dim.config_option),
            hidden_dim=dim,
            layer_norm_first=layer_norm_first,
            dtype=dtype,
            linear_config_1=linear_1_config,
            linear_config_2=linear_2_config,
            layer_norm=layer_norm_config,
            activation=ffn_activation_types
        )


class DenseFfnConfig(Config):

    def __init__(self,
                 intermedia_layers: list[FfnConfig],
                 out_layer: FfnConfig,
                 dense_dtype: DenseDType):
        self.dense_dtype = dense_dtype
        self.out_layer = out_layer
        self.intermedia_layers = intermedia_layers

    @property
    def output_dim(self):
        return self.out_layer.output_dim

    @property
    def input_dim(self):
        return self.intermedia_layers[0].input_dim

    @classmethod
    @autowire_fn()
    def dense_ffn(cls,
                  config_type: ConfigType,
                  in_dim: InDimDense,
                  num_layers: NumLayersDenseFfn,
                  out_dim: OutDimDense,
                  dtype: DenseDType):
        intermediate_layers = cls._intermediate_layers(in_dim, num_layers, dtype.config_option)
        return DenseFfnConfig(
            intermediate_layers,
            FfnConfig.ffn_with_override(
                in_dim=InDimFfn(sum([i.output_dim.config_option for i in intermediate_layers]) + in_dim.config_option),
                dim=EmbeddingSizeFfn(out_dim.config_option), dtype=FfnDType(dtype.config_option)),
            dense_dtype=dtype
        )

    @classmethod
    def _intermediate_layers(cls, in_dim, num_layers, dtype):
        intermediate_layers = []
        current = in_dim.config_option
        for i in range(num_layers.config_option):
            intermediate_layers.append(FfnConfig.ffn_with_override(
                in_dim=InDimFfn(current),
                dim=EmbeddingSizeFfn(current + (in_dim.config_option * (i + 1))),
                dtype=FfnDType(dtype)
            ))
            current += (in_dim.config_option * (i + 1) + current)
        return intermediate_layers


# TODO: can put this in densenet
def create_dense_layers(config_type, in_dim, num_layers_ffn_dense, out_dim_dense, num_layers_dense):
    intermediate_layers = _intermediate_layers(config_type, in_dim, num_layers_ffn_dense, num_layers_dense)
    last_layer = DenseFfnConfig.dense_ffn(
        config_type,
        InDimDense(intermediate_layers[-1].output_dim.config_option),
        NumLayersDenseFfn(num_layers_ffn_dense),
        OutDimDense(out_dim_dense)
    )
    intermediate_layers.append(last_layer)
    return intermediate_layers


def _intermediate_layers(config_type, in_dim, num_layers_ffn_dense, num_layers):
    intermediate_layers = []
    prev = in_dim
    current = in_dim * 2
    for i in range(2, num_layers + 2):
        intermediate_layers.append(DenseFfnConfig.dense_ffn(config_type,
                                                            InDimDense(prev),
                                                            NumLayersDenseFfn(num_layers_ffn_dense),
                                                            OutDimDense(current)))
        prev = current
        current = in_dim * (i + 1)
    return intermediate_layers
