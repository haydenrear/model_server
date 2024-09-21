import abc
import enum


class NoArgsSerializable(abc.ABC):
    pass


class SerializableEnum(enum.Enum):
    @classmethod
    def value_of(cls, value: str):
        for i, name in cls.__members__.items():
            if i.lower() == value.lower():
                return name

