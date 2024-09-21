import abc
import asyncio
import typing

from drools_py.subscription.publisher import Publisher, AsyncPublisher, AsyncPublisherWithBackPressure
from drools_py.subscription.subscriber import Subscriber, T, AsyncSubscriber

MessageT = typing.TypeVar("MessageT")
ProcessedT = typing.TypeVar("ProcessedT")


class BaseProcessor:
    pass


class Processor(Subscriber[MessageT],
                Publisher[Subscriber[ProcessedT], ProcessedT],
                abc.ABC,
                typing.Generic[MessageT, ProcessedT]):
    """
    Processor has subscribers and publishers and accepts data from subscriber, performs "process" on the data, and
    publishes to the it's subscribers.
    """

    def __init__(self, process_batch: int = 1, buffer_size: int = 1000, replace_buffer_messages: bool = True):
        Publisher.__init__(self, replace_buffer_messages, buffer_size)
        Subscriber.__init__(self)
        self.process_batch = process_batch

    @abc.abstractmethod
    def process(self, message: MessageT) -> ProcessedT:
        pass

    @abc.abstractmethod
    def process_batch(self, message: list[MessageT]) -> list[ProcessedT]:
        pass

    def next(self, message: MessageT):
        """
        Perform processing of message and publish to it's subscribers.
        :param message:
        :return:
        """
        message: ProcessedT = self.process(message)
        return super().next(message)

    def next_value(self, subscription_message: MessageT):
        """
        This is getting next value from it's publishers.
        :param subscription_message:
        :return:
        """
        self.next(subscription_message)

    def next_values(self, subscription_message: list[MessageT]):
        """
        This is getting next value from it's publishers.
        :param subscription_message:
        :return:
        """
        for s in subscription_message:
            self.next(s)

    def has_next(self) -> bool:
        return not self.buffer.is_empty()

    def ready(self) -> bool:
        return not self.buffer.is_full()

    def is_ready_num_messages(self) -> int:
        return self.buffer.num_kernels - self.buffer.n_messages_available()


class AsyncProcessor(BaseProcessor,
                     AsyncSubscriber[MessageT],
                     AsyncPublisherWithBackPressure[AsyncSubscriber[ProcessedT], ProcessedT],
                     abc.ABC,
                     typing.Generic[MessageT, ProcessedT]):

    @abc.abstractmethod
    async def process(self, message: MessageT) -> typing.Optional[ProcessedT]:
        """
        :param message: The message to process.
        :return: Optional processed. Will flatten (won't call next with None).
        """
        pass

    @abc.abstractmethod
    async def process_batch(self, message: list[MessageT]) -> list[ProcessedT]:
        pass

    async def next(self, message: MessageT) -> bool:
        """
        Perform processing of message and publish to it's subscribers.
        :param message:
        :return:
        """
        message: ProcessedT = await self.process(message)
        if message is not None:
            next_value = await super().next(message)
            return next_value
        else:
            return False

    async def next_value(self, subscription_message: MessageT):
        """
        This is getting next value from it's publishers.
        :param subscription_message:
        :return:
        """
        return await self.next(subscription_message)

    async def next_values(self, subscription_message: list[MessageT]) -> list[bool]:
        """
        This is getting next value from it's publishers.
        :param subscription_message:
        :return:
        """
        return await asyncio.gather(*[self.next(s) for s in subscription_message])
