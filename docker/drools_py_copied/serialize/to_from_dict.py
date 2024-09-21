import abc
import json


class ToDict(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass


class FromDict(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, val: dict):
        pass


class ToFromJsonDict(ToDict, FromDict, abc.ABC):

    def serialize_to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def serialize_from_json(cls, from_value: str):
        cls.from_dict(json.loads(from_value))
