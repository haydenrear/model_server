import typing
from typing import Optional

import torch.nn.init

from codegen.generated_config_models import QuerySize, DropoutAttn, AttnIncludeBias, \
    AttnIncludeBiasKv, AttnAddZeroAttn, AttnBatchFirst, OptionalKeySize, OptionalValueSize, AttnDType
from drools_py.attn.attn_kernel_function import AttnKernelFunctionConfigFactory
from drools_py.attn.attn_types import AttnTypes, AttnTypesConfigOption
from drools_py.attn.custom_proba_multi_head_attn import CustomProbaMultiheadAttention
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import EmbeddingSize, NumAttnHeads
from drools_py.output_probabilities.output_probabilities import OutputProbabilitiesConfigFactory, \
    OutputProbabilitiesConfig
from drools_py.output_probabilities.output_probabilities_config_option import OutputProbabilitiesTypes, \
    OutputProbabilitiesTypeConfigOption
from drools_py.weight_init.weight_initialization import WeightInitializationConfigFactory
from python_di.configs.prototype import prototype_scope_bean, prototype_factory


@prototype_scope_bean()
class AttnCalcConfig(Config):

    @prototype_factory()
    def __init__(
            self,
            output_probabilities: AttnTypesConfigOption,
            proba_type: OutputProbabilitiesTypeConfigOption,
            embed_dim: QuerySize,
            n_heads: NumAttnHeads,
            dropout: DropoutAttn,
            bias: AttnIncludeBias,
            add_bias_kv: AttnIncludeBiasKv,
            add_zero_attn: AttnAddZeroAttn,
            kdim: OptionalKeySize,
            vdim: OptionalValueSize,
            batch_first: AttnBatchFirst,
            dtype: AttnDType,
            weight_initialization: WeightInitializationConfigFactory,
            attn_kernel: typing.Optional[AttnKernelFunctionConfigFactory] = None
    ):
        self.weight_initialization = weight_initialization
        self.attn_kernel = attn_kernel
        self.dtype = dtype
        self.batch_first = batch_first
        self.kdim = kdim
        self.vdim = vdim
        self.add_zero_attn = add_zero_attn
        self.add_bias_kv = add_bias_kv
        self.bias = bias
        self.dropout = dropout
        self.n_heads = n_heads
        self.proba_type = OutputProbabilitiesConfigFactory(OutputProbabilitiesConfig(proba_type)) \
            if not isinstance(proba_type, OutputProbabilitiesConfigFactory) else proba_type
        self.embed_dim = embed_dim
        self.output_probabilities = output_probabilities


@prototype_scope_bean()
class AttnCalcConfigFactory(ConfigFactory):

    @prototype_factory()
    def __init__(self, output_probabilities_config: AttnCalcConfig):
        super().__init__(output_probabilities_config)
        self.output_probabilities_config = output_probabilities_config

    def create(self, embed_dim: Optional[EmbeddingSize] = None, **kwargs):
        assert embed_dim is not None or self.output_probabilities_config.embed_dim is not None, \
            "Output features not provided."
        embed_dim, kdim, vdim = self.get_dims(embed_dim)
        attn_cal = None
        if self.output_probabilities_config.output_probabilities.config_option == AttnTypes.DotProduct:
            if (self.output_probabilities_config.proba_type.output_probabilities_config.output_probabilities
                    == OutputProbabilitiesTypes.Softmax):
                attn_cal = torch.nn.MultiheadAttention(embed_dim,
                                                       self.output_probabilities_config.n_heads.config_option,
                                                       self.output_probabilities_config.dropout.config_option,
                                                       self.output_probabilities_config.bias.config_option,
                                                       self.output_probabilities_config.add_bias_kv.config_option,
                                                       self.output_probabilities_config.add_zero_attn.config_option,
                                                       kdim,
                                                       vdim,
                                                       self.output_probabilities_config.batch_first.config_option,
                                                       dtype=self.output_probabilities_config.dtype.config_option)
            else:
                attn_cal = CustomProbaMultiheadAttention(embed_dim,
                                                         self.output_probabilities_config.n_heads.config_option,
                                                         self.output_probabilities_config.proba_type.create(embed_dim),
                                                         self.output_probabilities_config.dropout.config_option,
                                                         self.output_probabilities_config.bias.config_option,
                                                         self.output_probabilities_config.add_bias_kv.config_option,
                                                         self.output_probabilities_config.add_zero_attn.config_option,
                                                         kdim,
                                                         vdim,
                                                         self.output_probabilities_config.batch_first.config_option,
                                                         dtype=self.output_probabilities_config.dtype.config_option)
        elif self.output_probabilities_config.output_probabilities == AttnTypes.FourierAttention:
            assert self.output_probabilities_config.attn_kernel is not None
            from drools_py.attn.fourier_attn import FourierAttn
            attn_cal = FourierAttn(self.output_probabilities_config)
        elif self.output_probabilities_config.output_probabilities == AttnTypes.FlashAttention:
            raise NotImplementedError()
        elif self.output_probabilities_config.output_probabilities == AttnTypes.LinearAttention:
            raise NotImplementedError()

        assert attn_cal is not None, \
            f"{self.output_probabilities_config.output_probabilities} was not a valid attention type."

        if self.output_probabilities_config.weight_initialization is not None:
            out = self.output_probabilities_config.weight_initialization.create(**kwargs)
            # out(attn_cal)
        return attn_cal

    def get_dims(self, embed_dim):
        if embed_dim is None:
            embed_dim = self.output_probabilities_config.embed_dim.config_option
        else:
            embed_dim = embed_dim.config_option
        kdim = self.output_probabilities_config.kdim.config_option \
            if (self.output_probabilities_config.kdim
                and self.output_probabilities_config.kdim.config_option != -1) \
            else None
        vdim = self.output_probabilities_config.vdim.config_option \
            if (self.output_probabilities_config.vdim
                and self.output_probabilities_config.vdim.config_option != -1) \
            else None
        return embed_dim, kdim, vdim
