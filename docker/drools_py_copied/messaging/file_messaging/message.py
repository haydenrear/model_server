import abc
import dataclasses

from drools_py.serialize.to_from_dict import ToFromJsonDict
from drools_py.messaging.message import MetadataMessage


@dataclasses.dataclass(init=True)
class FileMetadataMessage(MetadataMessage, ToFromJsonDict, abc.ABC):
    paths: list[str]

    def to_dict(self) -> dict:
        d = MetadataMessage.to_dict(self)
        d["paths"] = self.paths
        return d

    @classmethod
    def from_dict(cls, val: dict):
        return FileMetadataMessage(val["timestamp"], val["message_id"], val["paths"])


