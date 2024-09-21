import abc
from typing import Optional


class KafkaSerializer(abc.ABC):

    @abc.abstractmethod
    def serialize(self, input_message) -> bytes:
        pass


class KafkaDeserializer(abc.ABC):

    @abc.abstractmethod
    def serialize(self, input_value: bytes) -> Optional:
        pass
