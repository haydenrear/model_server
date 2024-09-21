from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable

from drools_py.messaging.config_pros import ProducerConfigProps


class ProducerFuture(ABC):

    @abstractmethod
    def add_callback(self, callback, *args, **kwargs):
        pass

    @abstractmethod
    def wait(self, timeout_ms: int = -1):
        pass


class FileProducerFuture(ProducerFuture):

    def __init__(self):
        pass

    def add_callback(self, callback, *args, **kwargs):
        pass

    def wait(self, timeout_ms: int = -1):
        pass


class TestProducerFuture(ProducerFuture):

    def wait(self, timeout_ms: int = -1):
        pass

    def add_callback(self, callback, *args, **kwargs):
        print("added callback!")
        pass


class Producer(ABC):

    def __init__(self, producer_config: ProducerConfigProps):
        self.producer_config: ProducerConfigProps = producer_config

    def send(self, topic: str, key: str, message: bytes,
             callback, headers: Optional[Dict[str, str]] = None,
             timestamp: Optional[int] = None) -> ProducerFuture:
        pass


    def send_callable(self,
                      topic: str,
                      key: str,
                      write_callable: Callable,
                      headers: Optional[Dict[str, str]] = None,
                      timestamp: Optional[int] = None,
                      metadata_message: dict = None) -> ProducerFuture:
        pass

    def flush(self):
        pass


class TestProducer(Producer):

    def flush(self):
        pass

    def __init__(self):
        super().__init__()

    def send(self, topic: str, key: str, message: bytes, callback,
             headers: Optional[Dict[str, str]] = None,
             timestamp: Optional[int] = None) -> ProducerFuture:
        return TestProducerFuture()
