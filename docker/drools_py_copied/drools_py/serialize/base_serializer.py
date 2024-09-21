import abc
import typing

import injector


ToSerializeT = typing.TypeVar("ToSerializeT")


class BaseSerializer(typing.Generic[ToSerializeT], abc.ABC):
    pass


class Serializers:
    @injector.inject
    def __init__(self, base_serializers: typing.List[BaseSerializer]):
        self.base_serializers = base_serializers


class DeSerializers:
    @injector.inject
    def __init__(self, base_serializers: typing.List[BaseSerializer]):
        self.base_serializers = base_serializers

    def primitive_ser(self):
        from drools_py.serialize.serializer import PrimitiveSerializer
        for i in self.base_serializers:
            if isinstance(i, PrimitiveSerializer):
                return i

    def dict_ser(self):
        from drools_py.serialize.serializer import DictSerializer
        for i in self.base_serializers:
            if isinstance(i, DictSerializer):
                return i
