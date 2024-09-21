from typing import Optional

import torch


class AttentionMask:
    def __init__(self, attn_mask: torch.Tensor):
        """
        Attention mask in the format of (N, S), where N is the batch size and S is the sequence
        :param attn_mask:
        """
        self.attn_mask = attn_mask

    def tch_key_padding_mask(self) -> torch.Tensor:
        pass

    def tch_src_mask(self) -> torch.Tensor:
        pass

    def get_mask_for(self, matching: torch.Tensor, padding_value: Optional[bool] = None) -> torch.Tensor:
        """
        Reduces or pads the mask according to this tensor. The default value for when the tensor size exceeds the mask
        size is to use the last value of the mask and extend it, however this padding value can be set using the
        padding_value param.
        :param matching:
        :param padding_value:
        :return:
        """
        pass


