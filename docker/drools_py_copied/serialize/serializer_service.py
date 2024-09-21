import os

import injector

from drools_py.serialize.base_serializer import BaseSerializer
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from drools_py.serialize.serializer_properties import SerializerProperties


class SerializerService:
    @injector.inject
    def __init__(self, properties: SerializerProperties, serializer: BaseSerializer):
        self.properties = properties
        self.serializer = serializer

    def serialize(self, to_serialize):
        return self.serializer.delegate_serialize(to_serialize)

    def de_serialize(self, to_deserialize):
        return self.serializer.delegate_de_serialize(to_deserialize)


@autowire_fn()
def serialize(to_serialize, serializer_service: SerializerService):
    return serializer_service.serialize(to_serialize)


@autowire_fn()
def de_serialize(to_serialize, serializer_service: SerializerService):
    return serializer_service.de_serialize(to_serialize)
