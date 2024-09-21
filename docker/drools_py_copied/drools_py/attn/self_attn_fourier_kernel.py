import math
from typing import Tuple, Optional

import torch
from drools_py.configs.config_models import EmbeddingSize
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.masking import pad_add_end_to_match

from drools_py.configs.config import ConfigType
from drools_py.layer_normalization.complex_norm import normalize_complex, ComplexLayerNorm

from drools_py.torch_utils.torch_prov_mod_configs import LayerNormConfig

from attn.attn_config import AttnConfig
from attn.attn_module import MultiHeadAttnModule
from codegen.generated_config_models import InDimAttnKernelConfig, RecursiveCatEmbeddingSize, \
    NumLayersRollingDecoderFourierKernel, NumAttnHeadsRecursiveCat, QuerySize, SequenceLengthRecursiveCat
from drools_py.attn.attn_kernel import AttnKernelTypesConfigOption, KernelTypes
from drools_py.attn.attn_kernel_config import AttnKernelConfig
from drools_py.attn.base_fourier_attn_kernel import BaseFourierAttnKernel
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_util.torch_utils.complex_torch import complex_to_2d, to_complex_from_2d, complex_to_2d_2d, \
    complex_boltzmann_prob
from python_util.torch_utils.masking import merge_masks_into_attn_mask
from python_util.torch_utils.pytorch_util import create_torch_size_log, does_tensor_have_nan
from test_framework.assertions.assert_all import Assert




class SelfAttnFourierKernelConfig(AttnKernelConfig):
    def __init__(self,
                 in_dim: InDimAttnKernelConfig,
                 imag_sa: AttnConfig,
                 real_sa: AttnConfig,
                 imag_real_ca: AttnConfig,
                 real_imag_ca: AttnConfig,
                 out_layer: AttnConfig,
                 ln: LayerNormConfig,
                 imag_real_ca_ln: LayerNormConfig,
                 real_imag_ca_ln: LayerNormConfig,
                 real_sa_ln: LayerNormConfig,
                 imag_sa_ln: LayerNormConfig,
                 seq_len: SequenceLengthRecursiveCat,
                 num_attn_heads: NumAttnHeadsRecursiveCat):
        super().__init__(AttnKernelTypesConfigOption(KernelTypes.SelfAttn), in_dim)
        self.real_imag_ca_ln = real_imag_ca_ln
        self.imag_sa_ln = imag_sa_ln
        self.real_sa_ln = real_sa_ln
        self.imag_real_ca_ln = imag_real_ca_ln
        self.real_imag_ca = real_imag_ca
        self.imag_real_ca = imag_real_ca
        self.real_sa = real_sa
        self.num_attn_heads = num_attn_heads
        self.seq_len = seq_len
        self.ln = ln
        self.out_layer = out_layer
        self.imag_sa = imag_sa

    @classmethod
    @autowire_fn()
    def self_attn_fourier_kernel_config(cls,
                                        config_type: ConfigType,
                                        recursive_cat_embed: RecursiveCatEmbeddingSize,
                                        seq_len: SequenceLengthRecursiveCat,
                                        num_attn_heads: NumAttnHeadsRecursiveCat):
        fourier_query = QuerySize(seq_len.config_option * num_attn_heads.config_option)
        final_fourier_query = QuerySize(seq_len.config_option * num_attn_heads.config_option * 2)
        return SelfAttnFourierKernelConfig(InDimAttnKernelConfig(recursive_cat_embed.config_option),
                                           AttnConfig.attn_config_override(
                                               config_type, num_attn_heads, fourier_query
                                           ),
                                           AttnConfig.attn_config_override(
                                               config_type, num_attn_heads, fourier_query
                                           ),
                                           AttnConfig.attn_config_override(
                                               config_type, num_attn_heads, fourier_query
                                           ),
                                           AttnConfig.attn_config_override(
                                               config_type, num_attn_heads,
                                               fourier_query),
                                           AttnConfig.attn_config_override(
                                               config_type, num_attn_heads, final_fourier_query
                                           ),
                                           LayerNormConfig.layer_norm_config_dim_override(
                                               config_type, dim=EmbeddingSize(seq_len.config_option * 2)),
                                           LayerNormConfig.layer_norm_config_dim_override(
                                               config_type, dim=EmbeddingSize(seq_len.config_option)),
                                           LayerNormConfig.layer_norm_config_dim_override(
                                               config_type, dim=EmbeddingSize(seq_len.config_option)),
                                           LayerNormConfig.layer_norm_config_dim_override(
                                               config_type, dim=EmbeddingSize(seq_len.config_option)),
                                           LayerNormConfig.layer_norm_config_dim_override(
                                               config_type, dim=EmbeddingSize(seq_len.config_option)),
                                           seq_len, num_attn_heads
                                           )


class SelfAttnFourierKernel(BaseFourierAttnKernel):
    def __init__(self, self_attn_kernel: SelfAttnFourierKernelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = self_attn_kernel
        # self.real_sa = MultiHeadAttnModule(self_attn_kernel.real_sa)
        # self.imag_sa = MultiHeadAttnModule(self_attn_kernel.imag_sa)
        # self.real_imag_ca = MultiHeadAttnModule(self_attn_kernel.imag_real_ca)
        # self.imag_real_ca = MultiHeadAttnModule(self_attn_kernel.real_imag_ca)
        # self.last = MultiHeadAttnModule(self_attn_kernel.out_layer)
        # self.ln = self_attn_kernel.ln.to_layer_norm()
        # self.real_sa_ln = self_attn_kernel.real_sa_ln.to_layer_norm()
        # self.imag_sa_ln = self_attn_kernel.imag_sa_ln.to_layer_norm()
        self.scale = self_attn_kernel.ln.size.config_option
        # self.real_imag_ca_ln = self_attn_kernel.imag_real_ca_ln.to_layer_norm()
        # self.imag_real_ca_ln = self_attn_kernel.real_imag_ca_ln.to_layer_norm()
        self.complex_norm = ComplexLayerNorm([self_attn_kernel.ln.size.config_option // 2])
        # self.out_complex_norm = ComplexLayerNorm([self_attn_kernel.ln.size.config_option // 2])

    def forward(self,
                input_value: torch.Tensor,
                return_kernels: bool = False,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        # TODO: use convolution here instead, (make use of convolution theorem)
        :param input_value:
        :param return_kernels:
        :param tgt_mask:
        :return:
        """
        Assert.debug_deferred(lambda: not does_tensor_have_nan(input_value),
                              lambda: f"{input_value} is is result from fourier attention, which has NAN.")

        # input_value = self.complex_norm(input_value)

        # src = input_value.shape[-1]
        # tgt = input_value.shape[-2]
        #
        # s = [i for i in input_value.shape[:-2]]
        # s.append(self.config.seq_len)
        # s.append(self.config.seq_len)
        # input_value = pad_add_end_to_match(s, input_value)

        # if tgt_mask is not None:
        #     tgt_mask = pad_add_end_to_match([tgt_mask.shape[0], self.config.seq_len], tgt_mask)
        #     tgt_mask = merge_masks_into_attn_mask(tgt_mask, tgt_mask,
        #                                           self.config.num_attn_heads.config_option)

        # start_shape = [i for i in input_value.shape]
        #
        # out_value = input_value.transpose(1, 2).reshape(
        #     input_value.shape[0],
        #     self.config.seq_len.config_option,
        #     self.config.seq_len.config_option * self.config.num_attn_heads.config_option).transpose(0, 1)

        # input_value = complex_to_2d_2d(out_value)

        # real_part = input_value[..., 0]
        # imag_part = input_value[..., 1]

        # LoggerFacade.info(f"Performing fourier self attn kernel with real: {create_torch_size_log(real_part)} and"
        #                   f"imaginary: {create_torch_size_log(imag_part)}")

        # imag_part = torch.nn.functional.log_softmax(imag_part, dim=-1)
        # real_part = torch.nn.functional.log_softmax(real_part, dim=-1)

        # real_part = self.real_sa_ln(self.real_sa(real_part, real_part, attn_mask=tgt_mask) + real_part)
        # imag_part = self.imag_sa_ln(self.imag_sa(imag_part, imag_part, attn_mask=tgt_mask) + imag_part)

        # imag_part = torch.nn.functional.log_softmax(imag_part, dim=-1)
        # real_part = torch.nn.functional.log_softmax(real_part, dim=-1)

        # Assert.debug_deferred(lambda: not does_tensor_have_nan(imag_part),
        #                       lambda: f"{imag_part} is is result from fourier attention, which has NAN.")
        # Assert.debug_deferred(lambda: not does_tensor_have_nan(real_part),
        #                       lambda: f"{real_part} is is result from fourier attention, which has NAN.")

        # imag_part = self.imag_real_ca_ln(self.real_imag_ca(real_part, imag_part, attn_mask=tgt_mask) + imag_part)
        # real_part = self.real_imag_ca_ln(self.imag_real_ca(imag_part, real_part, attn_mask=tgt_mask) + real_part)

        # Assert.debug_deferred(lambda: not does_tensor_have_nan(imag_part),
        #                       lambda: f"{imag_part} is is result from fourier attention, which has NAN.")
        # Assert.debug_deferred(lambda: not does_tensor_have_nan(real_part),
        #                       lambda: f"{real_part} is is result from fourier attention, which has NAN.")

        LoggerFacade.info(f"Performing fourier self attn kernel with input: {create_torch_size_log(input_value)}.")
        # like ... Boltzmann! #TODO: maybe can replace this calc as the proba and then simply input the complex tensor
        #                       in CustomProbaAttention, or use the sum 0-1 norm from quantum. Update to use log.
        #                       - maybe the energy could be adding them together and taking absolute value
        # transpose = (torch.stack([torch.nn.functional.softmax(imag_part, dim=-1),
        #                          torch.nn.functional.softmax(real_part, dim=-1)])
        #              .transpose(-1, 0).transpose(-2, -3))

        # LoggerFacade.info(f"Performing fourier self attn kernel with transposed: {create_torch_size_log(transpose)}.")

        # Assert.debug_deferred(lambda: not does_tensor_have_nan(transpose),
        #                       lambda: f"{transpose} is is result from fourier attention, which has NAN.")
        #
        # reshaped_input = self._reshape_input_d(transpose)
        #
        # LoggerFacade.info(f"Performing fourier self attn kernel after reshaped: "
        #                   f"{create_torch_size_log(reshaped_input)}.")
        #
        # reshaped_input *= (1 / math.sqrt(self.scale))

        # reshaped_input = self.ln(reshaped_input)

        # calced, attn_scores = self.last(reshaped_input, return_attn_scores=True, attn_mask=tgt_mask)

        # Assert.debug_deferred(lambda: not does_tensor_have_nan(calced),
        #                       lambda: f"{calced} is is result from fourier attention, which has NAN.")
        #
        # out_value = calced + reshaped_input
        # Assert.debug_deferred(lambda: not does_tensor_have_nan(out_value),
        #                       lambda: f"{out_value} is is result from fourier attention, which has NAN.")
        #
        # complex_out = to_complex_from_2d(out_value)
        # complex_out = complex_out.reshape(start_shape)
        #
        # complex_out = self.out_complex_norm(complex_out)
        #
        # Assert.debug_deferred(lambda: not does_tensor_have_nan(complex_out),
        #                       lambda: f"{complex_out} is is result from fourier attention, which has NAN.")

        # if return_kernels:
        return None, self.complex_norm(complex_boltzmann_prob(input_value, dim=-1)), None
        # else:
        #     return None, attn_scores, None

    def _reshape_input_d(self, input_value):
        new_shape = [i for i in input_value.shape[:-2]]
        new_shape.append(input_value.shape[-1] * input_value.shape[-2])
        reshaped_input = input_value.reshape(new_shape)
        return reshaped_input
