import abc
from typing import Tuple, Optional

import torch.nn


class BaseFourierAttnKernel(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self,
                input_value: torch.Tensor,
                return_kernels: bool = False,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pass
