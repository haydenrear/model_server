import abc
import dataclasses

from drools_py.serialize.to_from_dict import ToFromJsonDict


@dataclasses.dataclass(init=True)
class MetadataMessage(ToFromJsonDict, abc.ABC):
    timestamp: int
    message_id: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, val: dict):
        return MetadataMessage(val["timestamp"], val["message_id"])


@dataclasses.dataclass(init=True)
class DataMessage(abc.ABC):
    pass


class FileMessageBuilder(abc.ABC):

    @abc.abstractmethod
    def add_metadata(self, metadata: MetadataMessage):
        pass

    @abc.abstractmethod
    def add_message(self, message):
        pass

    @abc.abstractmethod
    def build(self):
        pass


class FileMessageBuilderFactory(abc.ABC):

    @abc.abstractmethod
    def create_message_builder(self) -> FileMessageBuilder:
        pass
