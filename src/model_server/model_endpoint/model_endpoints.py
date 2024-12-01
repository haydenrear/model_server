import abc
import typing

import torch

from metadata_extractor.messaging.reflectable_media_component import FromJsonClass, ToJsonClass
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope

OutT = typing.TypeVar("OutT", covariant=True, bound=ToJsonClass)
InT = typing.TypeVar("InT", covariant=True, bound=FromJsonClass)

class GenericModelEndpoint(abc.ABC, typing.Generic[InT, OutT]):
    """
    Any embedding to numpy array such as tokenization or ML embedding.
    """

    @abc.abstractmethod
    def do_model(self, input_data: InT) -> OutT:
        pass

    @property
    @abc.abstractmethod
    def endpoint(self) -> str:
        pass


class ModelEndpoint(typing.Generic[OutT], GenericModelEndpoint[dict[str, ...], OutT], abc.ABC):
    """
    Any embedding to numpy array such as tokenization or ML embedding.
    """

    @abc.abstractmethod
    def do_model(self, input_data: dict[str, ...]) -> OutT:
        pass

    @property
    @abc.abstractmethod
    def endpoint(self) -> str:
        pass



class PytorchModelEndpoint(GenericModelEndpoint[InT, OutT], torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def convert_to(self, input_data: InT) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def convert_from(self, input_data: torch.Tensor) -> OutT:
        pass

    @abc.abstractmethod
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        pass

    def do_model(self, input_data: InT) -> OutT:
        return self.convert_from(self(self.convert_to(input_data)))

