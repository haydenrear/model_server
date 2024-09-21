import torch

from drools_py.configs.config_models import NumAttnHeads
from python_util.torch_utils.masking import make_causal_mask, merge_masks_into_attn_mask


class AttnMaskDelegate:

    def __init__(self, src_mask: torch.Tensor, tgt_mask: torch.Tensor, n_heads: int = 1, do_heads: bool = True):
        self._src_mask = src_mask
        self._tgt_mask = tgt_mask
        self._merged_mask = merge_masks_into_attn_mask(src_mask, tgt_mask, n_heads)
        self._causal_mask = make_causal_mask(self._merged_mask)
        if not do_heads and self._merged_mask.shape[0] == 1:
            self._merged_mask = self._merged_mask.squeeze(0)
            self._causal_mask = self._causal_mask.squeeze(0)

    @property
    def src_mask(self):
        return self._src_mask

    @property
    def tgt_mask(self):
        return self._tgt_mask

    @property
    def causal_mask(self):
        return self._causal_mask

    @property
    def merged_mask(self):
        return self._merged_mask

    def merged_mask_matching(self, src: torch.Tensor, tgt: torch.Tensor):
        return self._merged_mask[:, :tgt.shape[0], :src.shape[0]]
