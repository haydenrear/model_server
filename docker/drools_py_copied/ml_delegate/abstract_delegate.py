import abc
import dataclasses
import typing

import torch

from python_util.logger.logger import LoggerFacade


class AbstractDelegateMetadata(abc.ABC):
    @abc.abstractmethod
    def key(self) -> str:
        pass


MetadataT = typing.TypeVar("MetadataT", bound=AbstractDelegateMetadata, covariant=True)


class AbstractDelegator(abc.ABC, typing.Generic[MetadataT]):

    @abc.abstractmethod
    def delegate(self, metadata: MetadataT, *args, **kwargs):
        pass

    @abc.abstractmethod
    def no_op(self, metadata: MetadataT, *args, **kwargs):
        pass

    @abc.abstractmethod
    def matches_delegate(self, metadata: MetadataT) -> bool:
        pass

    def do_delegate(self, metadata: MetadataT, *args, **kwargs):
        if self.matches_delegate(metadata):
            return self.delegate(metadata, *args, **kwargs)
        else:
            return self.no_op(metadata, *args, **kwargs)


DelegatorT = typing.TypeVar("DelegatorT", bound=AbstractDelegator, covariant=True)


class NoOpTensorDelegator(abc.ABC, AbstractDelegator):

    @abc.abstractmethod
    def delegate(self, metadata: MetadataT, in_data: torch.Tensor):
        pass

    @abc.abstractmethod
    def matches_delegate(self, metadata: MetadataT) -> bool:
        pass

    def no_op(self, metadata: MetadataT, in_data: torch.Tensor):
        return in_data



@dataclasses.dataclass(init=True)
class AbstractDelegate(typing.Generic[MetadataT]):
    """
    There are several operations that can be performed in many different places, such as before or after layer norm.
    So, call function in all places and only do it based on config, being the feature flag delegator.
    """
    start_register: AbstractDelegator
    finish_register: AbstractDelegator
    pre_call_register: AbstractDelegator
    call_register: AbstractDelegator
    post_call_register: AbstractDelegator

    def start(self, metadata: MetadataT, *args, **kwargs):
        return self.start_register.do_delegate(metadata, *args, **kwargs)

    def finish(self, metadata: MetadataT, *args, **kwargs):
        return self.finish_register.do_delegate(metadata, *args, **kwargs)

    def call(self, metadata: MetadataT, *args, **kwargs):
        pre = self.pre_call_register.do_delegate(metadata, *args, **kwargs)
        pre = self.call_register.do_delegate(pre, *args, **kwargs)
        return self.post_call_register.do_delegate(pre, *args, **kwargs)


