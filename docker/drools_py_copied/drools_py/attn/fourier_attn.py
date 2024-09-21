import math
import typing

import torch.nn
from torch import nn
from torch.fft import fft, ifft

import python_util.torch_utils.masking
from drools_py.attn.attn import AttnCalcConfig
from drools_py.attn.self_attn_fourier_kernel import SelfAttnFourierKernelConfig
from drools_py.layer_normalization.complex_norm import ComplexLayerNorm
from drools_py.mask.attn_mask_delegate import AttnMaskDelegate
from drools_py.output_probabilities.output_probabilities import complex_softmax
from python_util.torch_utils.complex_torch import ifft_to_real, complex_attn_mask


class FourierAttn(torch.nn.Module):

    def __init__(self,
                 attn_calc_config: AttnCalcConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.attn_kernel = attn_calc_config.attn_kernel.create()
        self.d_model = attn_calc_config.embed_dim.config_option
        self.num_heads = attn_calc_config.n_heads.config_option
        self.head_dim = self.d_model // self.num_heads
        self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.out_proj_weight = torch.nn.ParameterList([
            nn.Linear(self.d_model, self.d_model, dtype=torch.complex64)
            for _ in range(4)
        ])
        self.complex_norm = ComplexLayerNorm([self.d_model])
        self.in_complex_norm = ComplexLayerNorm([self.d_model])
        self.kernel_complex_norm = ComplexLayerNorm(
            [typing.cast(SelfAttnFourierKernelConfig, attn_calc_config.attn_kernel.attn_kernel_config)
             .seq_len.config_option]
        )

    def forward(self, query, key=None, value=None,
                attn_mask_delegate: typing.Optional[AttnMaskDelegate] = None,
                return_kernels: bool = False,
                multiply_by_kernels: bool = False,
                project_kernels_to_real: bool = False,
                return_outer_product: bool = False,
                return_freq: bool = False,
                causal: bool = False,
                do_ifft: bool = True):

        if len(query.shape) == 2:
            query = query.unsqueeze(1)
            batch_size = 1
        else:
            batch_size = query.size(1)

        if key is not None and len(key.shape) == 2:
            key = key.unsqueeze(1)
        if value is not None and len(value.shape) == 2:
            value = value.unsqueeze(1)

        if key is None:
            key = query
        if value is None:
            value = key

        # Project query, key, and value # TODO: make these complex

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, key_dim = key.shape
        v_len, _, v_dim = value.shape

        q_ft = self._to_frequency(query)
        k_ft = self._to_frequency(key)
        v_ft = self._to_frequency(value)

        q_ft = self.q_proj_weight(q_ft)
        k_ft = self.k_proj_weight(k_ft)
        v_ft = self.v_proj_weight(v_ft)

        q_ft = self._expand_attn_heads(batch_size, q_ft)
        k_ft = self._expand_attn_heads(batch_size, k_ft)
        v_ft = self._expand_attn_heads(batch_size, v_ft)

        q_ft = q_ft.transpose(0, 1).reshape(tgt_len, batch_size * self.num_heads, embed_dim)
        k_ft = k_ft.transpose(0, 1).reshape(src_len, batch_size * self.num_heads, key_dim)
        v_ft = v_ft.transpose(0, 1).reshape(v_len, batch_size * self.num_heads, v_dim)

        q_ft = self.in_complex_norm(q_ft)
        k_ft = self.in_complex_norm(k_ft)

        out_mask, tgt_mask = self._get_create_mask(attn_mask_delegate, causal)

        outer_prod = torch.baddbmm(out_mask, q_ft.transpose(0, 1), k_ft.conj().transpose(0, 1).transpose(-2, -1))
        outer_prod *= (1 / math.sqrt(self.d_model))

        scores = complex_softmax(outer_prod, dim=-1)

        # v_ft = [2, 128, 10]
        v_ft = self.in_complex_norm(v_ft)

        # out = [2, 10, 128], out_fft = [2, 10, 128] # TODO: try scaling ? out_prod_scaled = out_prod / math.sqrt(v_ft.shape[-1])
        out_fft = torch.matmul(scores, v_ft.transpose(0, 1).reshape(batch_size, self.num_heads, v_len, v_dim))
        out_fft = torch.matmul(out_mask, out_fft)

        # reshape to [attn_heads, seq_len, batch_size, dim]
        out_fft = out_fft.permute(1, 2, 0, 3)
        out_fft = self.complex_norm(out_fft)

        if do_ifft:
            out = ifft(out_fft, dim=-1)
            out = ifft_to_real(out)
        else:
            out = out_fft

        # Reshape and project back
        if return_outer_product:
            if return_freq:
                return out, scores, _, outer_prod, out_fft
            else:
                return out, scores, _, outer_prod, None
        elif return_freq:
            return out, scores, _, None, out_fft
        else:
            return out, scores, _, None, None

    def _expand_attn_heads(self, batch_size, q_ft):
        if len(q_ft.shape) <= 3:
            q_ft = q_ft.expand(self.num_heads, q_ft.size(0), batch_size, q_ft.size(-1))
        return q_ft

    def _to_frequency(self, query):
        if not query.dtype == torch.complex64:
            q_ft = fft(query, dim=-1)
        else:
            q_ft = query
        return q_ft

    def _get_create_mask(self, attn_mask_delegate, causal):
        src_mask = attn_mask_delegate.src_mask if attn_mask_delegate is not None else None
        tgt_mask = attn_mask_delegate.tgt_mask if attn_mask_delegate is not None else None
        mask_delegate = AttnMaskDelegate(src_mask, tgt_mask)
        if causal:
            out_mask = mask_delegate.causal_mask
        else:
            out_mask = mask_delegate.merged_mask
        out_mask = complex_attn_mask(out_mask)
        return out_mask, tgt_mask

    def _apply_v_attn_mask(self, src_attn_mask, src_mask, v_ft):
        if src_mask is not None:
            v_ft = python_util.torch_utils.masking.apply_mask_as_value(v_ft, src_attn_mask.transpose(0, 1))
        return v_ft

    def _apply_outer_prod_attn_mask(self, batch_size, outer_prod, tgt_mask):
        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(1).expand(batch_size, self.num_heads, -1)
            outer_prod = python_util.torch_utils.masking.apply_mask_as_value(outer_prod, tgt_mask.transpose(0, 1))
        return outer_prod

    def _apply_qk_attn_mask(self, tgt_attn_mask, src_attn_mask, k_ft, q_ft):
        if tgt_attn_mask is not None:
            q_ft = python_util.torch_utils.masking.apply_mask_as_value(q_ft, tgt_attn_mask)
        if src_attn_mask is not None:
            k_ft = python_util.torch_utils.masking.apply_mask_as_value(k_ft, src_attn_mask)
        return k_ft, q_ft

    def _get_reshape_attn_masks(self, batch_size, src_mask, tgt_mask):
        attn_mask = None
        src_attn_mask = None
        if tgt_mask is not None:
            attn_mask = tgt_mask.expand(self.num_heads, batch_size, tgt_mask.shape[1]).reshape(
                self.num_heads * batch_size, -1)
        if src_mask is not None:
            src_attn_mask = src_mask.expand(self.num_heads, batch_size, src_mask.shape[1]).reshape(
                self.num_heads * batch_size, -1)
        return attn_mask, src_attn_mask
