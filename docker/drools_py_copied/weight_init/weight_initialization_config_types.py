import abc
import enum

import torch

from drools_py.attn.custom_proba_multi_head_attn import CustomProbaMultiheadAttention
from drools_py.configs.config_models import EnumConfigOption, ConfigOption
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.pytorch_util import copy_tensor_to


class WeightInitializationTypes(EnumConfigOption):
    XavierUniform = enum.auto()
    XavierNormal = enum.auto()
    Zero = enum.auto()
    Ones = enum.auto()
    Random = enum.auto()
    Uniform = enum.auto()
    KaimingUniform = enum.auto()
    KaimingNormal = enum.auto()
    LeCun = enum.auto()
    Orthogonal = enum.auto()
    Normal = enum.auto()
    WeightTying = enum.auto()
    WeightInitializationConsumer = enum.auto()
    ComplexFrequencyBands = enum.auto()
    He = enum.auto()


class WeightInitializationTypesConsumer(abc.ABC):
    @abc.abstractmethod
    def consume(self, in_module: torch.nn.Module, **kwargs):
        pass


class AttnWeightInitializationConsumer(WeightInitializationTypesConsumer):

    def __init__(self, key_factory, query_factory, value_factory, out_proj_factory,
                 in_proj_bias_factory, out_proj_bias_factory, bias_k_factory, bias_v_factory):
        from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory
        self.bias_k_factory: WeightInitializationConfigFactory = bias_k_factory
        self.out_proj_bias_factory: WeightInitializationConfigFactory = out_proj_bias_factory
        self.in_proj_bias_factory: WeightInitializationConfigFactory = in_proj_bias_factory
        self.bias_v_factory: WeightInitializationConfigFactory = bias_v_factory
        self.out_proj_factory: WeightInitializationConfigFactory = out_proj_factory
        self.value_factory: WeightInitializationConfigFactory = value_factory
        self.query_factory: WeightInitializationConfigFactory = query_factory
        self.key_factory: WeightInitializationConfigFactory = key_factory

    @classmethod
    def same_key_same_bias(cls, weights: WeightInitializationTypes, bias: WeightInitializationTypes):
        return AttnWeightInitializationConsumer(
            cls.get_factory(weights), cls.get_factory(weights), cls.get_factory(weights), cls.get_factory(weights),
            cls.get_factory(bias), cls.get_factory(bias), cls.get_factory(bias), cls.get_factory(bias),
        )

    @classmethod
    def get_factory(cls, w: WeightInitializationTypes):
        from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory
        return WeightInitializationConfigFactory.weight_initialization_config_factory(
            WeightInitializationTypesConfigOption(w))

    def consume(self, attn_module, **kwargs):
        from drools_py.attn.fourier_attn import FourierAttn
        if isinstance(attn_module, CustomProbaMultiheadAttention):
            self._initialize_attn_module_weights(attn_module)
        elif isinstance(attn_module, torch.nn.MultiheadAttention):
            self._initialize_attn_module_weights(attn_module)
        elif isinstance(attn_module, FourierAttn):
            self._initialize_fourier_attn_module_weights(attn_module)
        else:
            raise ValueError(f"Could not initialize attention module: {type(attn_module)}.")

    def _initialize_fourier_attn_module_weights(self, attn_module):
        from drools_py.attn.fourier_attn import FourierAttn
        attn_module: FourierAttn = attn_module
        if attn_module.k_proj_weight is not None:
            attn_module.k_proj_weight.weight = self.key_factory.create()(attn_module.k_proj_weight.weight)
        if attn_module.v_proj_weight is not None:
            attn_module.v_proj_weight.weight = self.value_factory.create()(attn_module.v_proj_weight.weight)
        if attn_module.q_proj_weight is not None:
            attn_module.q_proj_weight.weight = self.query_factory.create()(attn_module.q_proj_weight.weight)
        if attn_module.out_proj is not None:
            attn_module.out_proj.weight = self.out_proj_factory.create()(attn_module.out_proj.weight)
        if attn_module.k_proj_weight.bias is not None:
            attn_module.k_proj_weight.bias = self.bias_k_factory.create()(attn_module.k_proj_weight.bias)
        if attn_module.v_proj_weight.bias is not None:
            attn_module.v_proj_weight.bias = self.bias_v_factory.create()(attn_module.v_proj_weight.bias)
        if attn_module.in_proj_weight.bias is not None:
            attn_module.in_proj_weight.bias = self.in_proj_bias_factory.create()(attn_module.in_proj_weight.bias)

    def _initialize_attn_module_weights(self, attn_module):
        if attn_module.in_proj_weight is not None:
            attn_module.in_proj_weight = self.query_factory.create()(attn_module.in_proj_weight)
        if attn_module.k_proj_weight is not None:
            attn_module.k_proj_weight = self.key_factory.create()(attn_module.k_proj_weight)
        if attn_module.v_proj_weight is not None:
            attn_module.v_proj_weight = self.value_factory.create()(attn_module.v_proj_weight)
        if attn_module.q_proj_weight is not None:
            attn_module.q_proj_weight = self.query_factory.create()(attn_module.q_proj_weight)
        if attn_module.out_proj is not None:
            next_value = self.out_proj_factory.create()(
                attn_module.out_proj.weight if hasattr(attn_module.out_proj, 'weight')
                else attn_module.out_proj)
            if hasattr(attn_module.out_proj, 'weight'):
                attn_module.out_proj.weight = next_value
            else:
                attn_module.out_proj = next_value
        if attn_module.bias_k is not None:
            attn_module.bias_k = self.bias_k_factory.create()(attn_module.bias_k)
        if attn_module.bias_v is not None:
            attn_module.bias_v = self.bias_v_factory.create()(attn_module.bias_v)
        if attn_module.in_proj_bias is not None:
            attn_module.in_proj_bias = self.in_proj_bias_factory.create()(attn_module.in_proj_bias)
        if (attn_module.out_proj is not None
                and hasattr(attn_module.out_proj, 'bias')
                and attn_module.out_proj.bias is not None):
            next_value = self.out_proj_bias_factory.create()(attn_module.out_proj.bias)
            if hasattr(attn_module.out_proj, 'bias'):
                attn_module.out_proj.bias = next_value
            else:
                attn_module.out_proj = next_value


class ComplexWeightInitConsumer(WeightInitializationTypesConsumer):

    def __init__(self, real_weight_init, imag_weight_init=None):
        self.real_weight_init = real_weight_init
        self.imag_weight_init = imag_weight_init

    def __call__(self, *args, **kwargs):
        return self.consume(*args, **kwargs)

    def consume(self, attn_module: torch.nn.Module, **kwargs):
        if self.imag_weight_init is not None:
            attn_module.imag = self.imag_weight_init.create(**kwargs)(attn_module.imag)
            attn_module.real = self.real_weight_init.create(**kwargs)(attn_module.real)
        else:
            attn_module = self.real_weight_init.create()(attn_module, **kwargs)

        return attn_module


class ComplexAttnWeightInitConsumer(AttnWeightInitializationConsumer):

    def __init__(self, complex_part: AttnWeightInitializationConsumer):
        super().__init__(complex_part.key_factory, complex_part.query_factory, complex_part.value_factory,
                         complex_part.out_proj_factory, complex_part.in_proj_bias_factory,
                         complex_part.out_proj_bias_factory, complex_part.bias_k_factory, complex_part.bias_v_factory)

    @classmethod
    def same_key_same_bias(cls,
                           real_weights: ComplexWeightInitConsumer,
                           real_bias: ComplexWeightInitConsumer):
        return ComplexAttnWeightInitConsumer(AttnWeightInitializationConsumer(
            cls.get_factory(real_weights),
            cls.get_factory(real_weights),
            cls.get_factory(real_weights),
            cls.get_factory(real_weights),
            cls.get_factory(real_bias),
            cls.get_factory(real_bias),
            cls.get_factory(real_bias),
            cls.get_factory(real_bias),
        ))

    @classmethod
    def get_factory(cls, w: ComplexWeightInitConsumer):
        from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory
        return WeightInitializationConfigFactory.weight_initialization_config_factory(
            weight_types_config=WeightInitializationTypesConfigOption(WeightInitializationTypes.WeightInitializationConsumer),
            weights_provider=w)



class WeightTyingWeightInitializationConsumer(WeightInitializationTypesConsumer):

    def __init__(self, weight: torch.Tensor):
        self.weight: torch.Tensor = weight

    def consume(self, in_module: torch.Tensor, **kwargs):
        LoggerFacade.debug(f"Weight tying initialization strategy being executed. Updating weights of size "
                           f"{in_module.shape} to {self.weight.shape}")
        copy_tensor_to(in_module, self.weight)
        return in_module


class WeightInitializationTypesConfigOption(ConfigOption[WeightInitializationTypes]):
    def __init__(self, config_option: WeightInitializationTypes = WeightInitializationTypes.KaimingNormal):
        super().__init__(config_option)
        self.config_option = config_option


class WeightInitializationTypesLnConfigOption(WeightInitializationTypesConfigOption):
    def __init__(self, config_option: WeightInitializationTypes = WeightInitializationTypes.Ones):
        super().__init__(config_option)
        self.config_option = config_option


class WeightInitializationTypesLnBiasConfigOption(WeightInitializationTypesLnConfigOption):
    def __init__(self, config_option: WeightInitializationTypes = WeightInitializationTypes.Zero):
        super().__init__(config_option)
        self.config_option = config_option
