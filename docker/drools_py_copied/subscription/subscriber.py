import abc
import typing
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class Subscriber(ABC, Generic[T]):

    @abstractmethod
    def next_value(self, subscription_message: T) -> bool:
        pass

    def next_values(self, subscription_message: [T]) -> list[bool]:
        return [
            self.next_value(s) for s in subscription_message
        ]

    def ready(self) -> bool:
        return True

    def is_ready_num_messages(self) -> int:
        return 1


MultiTypeT = TypeVar("MultiTypeT")


class MultiSubscriber(Subscriber, abc.ABC, typing.Generic[MultiTypeT]):
    @abc.abstractmethod
    def is_ready_for(self, ty: MultiTypeT):
        pass


class AsyncSubscriber(Subscriber[T], abc.ABC, Generic[T]):

    @abstractmethod
    async def next_value(self, subscription_message: T) -> bool:
        pass

    @abstractmethod
    async def next_values(self, subscription_message: list[T]) -> list[bool]:
        pass

    def ready(self) -> bool:
        return True

    def is_ready_num_messages(self) -> int:
        return 1


class AsyncSubscriberWithDynamicBackpressure(AsyncSubscriber[T], abc.ABC, typing.Generic[T]):

    @abc.abstractmethod
    def ready_for(self, value: T) -> bool:
        pass
