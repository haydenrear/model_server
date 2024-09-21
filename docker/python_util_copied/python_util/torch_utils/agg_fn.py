import abc
from typing import Optional

import torch


class AggregationModule(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, in_values: list[torch.Tensor], **kwargs) -> torch.Tensor:
        pass


class DecoderAggFn(abc.ABC):

    @abc.abstractmethod
    def do_agg(self, x, dim: Optional[int] = None, keep_dim: bool = False) -> torch.Tensor:
        pass


class MeanAgg(DecoderAggFn, AggregationModule):

    def do_agg(self, x, dim: Optional[int] = None, keep_dim: bool = False) -> torch.Tensor:
        return torch.mean(x, dim=dim if dim else 0, keepdim=keep_dim)

    def forward(self, in_values: list[torch.Tensor], **kwargs) -> torch.Tensor:
        return self.do_agg(in_values, **kwargs)
