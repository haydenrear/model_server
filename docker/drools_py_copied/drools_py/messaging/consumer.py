import abc
import asyncio
import typing
from typing import Optional

from drools_py.messaging.config_pros import ConsumerConfigProps
from python_util.logger.logger import LoggerFacade


class ConsumerRecord:
    def __init__(self, key: bytes, value: bytes,
                 header: Optional[dict[str, str]] = None,
                 timestamp: Optional[int] = None):
        self.timestamp = timestamp
        self.value = value
        self.key = key
        self.header = header


ConsumerMessageT = typing.TypeVar("ConsumerMessageT")


class Consumer(abc.ABC, typing.Generic[ConsumerMessageT]):

    def __init__(self, consumer_config_props: ConsumerConfigProps):
        self.consumer_config_props = consumer_config_props
        self.cancel_consumer: asyncio.Event = asyncio.Event()

    @abc.abstractmethod
    def initialize(self, topics: list[str], partitions: list[int]):
        pass

    @abc.abstractmethod
    def read_next_messages(self, num_read: int, timeout: int) -> list[ConsumerMessageT]:
        pass

    @abc.abstractmethod
    def consume_message(self, message: ConsumerMessageT):
        pass

    @property
    def timeout(self) -> int:
        return self.consumer_config_props.timeout

    @property
    def max_batch(self) -> int:
        return self.consumer_config_props.max_batch

    def cancel(self):
        self.cancel_consumer.set()

    def do_run_consumer(self) -> bool:
        return not self.cancel_consumer.is_set()

    def run_consumer(self):
        while self.do_run_consumer():
            self.step_consumer_messages()

    def step_consumer_messages(self):
        next_messages = self.read_next_messages(self.max_batch, self.timeout)
        if next_messages is not None and len(next_messages) != 0:
            for n in next_messages:
                self.consume_message(n)


class AsyncConsumer(abc.ABC, typing.Generic[ConsumerMessageT]):

    def __init__(self, consumer_config_props: ConsumerConfigProps):
        self.consumer_config_props = consumer_config_props
        self.cancel_consumer: asyncio.Event = asyncio.Event()

    @abc.abstractmethod
    def initialize(self, topics: list[str], partitions: list[int]):
        pass

    @abc.abstractmethod
    def read_next_messages(self, num_read: int, timeout: int) -> list[ConsumerMessageT]:
        pass

    @property
    def timeout(self) -> int:
        return self.consumer_config_props.timeout

    @property
    def max_batch(self) -> int:
        return self.consumer_config_props.max_batch

    def cancel(self):
        self.cancel_consumer.set()

    def do_run_consumer(self) -> bool:
        return not self.cancel_consumer.is_set()

    @abc.abstractmethod
    async def consume_message(self, message: ConsumerMessageT):
        pass

    async def run_consumer(self):
        while self.do_run_consumer():
            await self.step_consumer_messages()

    async def step_consumer_messages(self):
        LoggerFacade.debug(f"Stepping consumer publisher {self.__class__.__name__}.")
        next_messages = self.read_next_messages(self.max_batch, self.timeout)
        if next_messages is not None and len(next_messages) != 0:
            for n in next_messages:
                await self.consume_message(n)

