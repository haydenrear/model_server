import typing

import injector as injector
from injector import Binder

from drools_py.configs.config import ConfigType
from drools_py.serialize.base_serializer import BaseSerializer, Serializers, DeSerializers
from drools_py.serialize.serializer import Serializer, NoArgsTorchModuleSerializer, ConfigOptionSerializer, \
    ConfigSerializer, SerializableEnumSerializer, NoArgsCreatorSerialize, ReflectingSerializer, \
    SqlAlchemyStringSerializer, TupleSerializer, SetSerializer, TorchTensorSerializer, ListSerializer, DictSerializer, \
    PrimitiveSerializer
from drools_py.serialize.serializer_properties import SerializerProperties
from drools_py.serialize.serializer_service import SerializerService
from python_di.inject.context_builder.inject_ctx import inject_context_di
from python_di.inject.injector_provider import InjectionContextInjector
from python_di.configs.enable_configuration_properties import enable_configuration_properties
from python_di.inject.reflectable_ctx import bind_multi_bind

T = typing.TypeVar("T")


class SerializationCtx(injector.Module):
    def configure(self, binder: Binder) -> None:
        binder.bind(SerializerService, SerializerService, scope=injector.singleton)
        bind_multi_bind([
            NoArgsTorchModuleSerializer,
            ConfigOptionSerializer,
            ConfigSerializer,
            SerializableEnumSerializer,
            NoArgsCreatorSerialize,
            ReflectingSerializer,
            SqlAlchemyStringSerializer,
            TupleSerializer,
            SetSerializer,
            ListSerializer,
            TorchTensorSerializer,
            DictSerializer,
            PrimitiveSerializer,
        ], binder, typing.List[BaseSerializer])

        binder.bind(BaseSerializer, Serializer, scope=injector.singleton)
        binder.bind(Serializers, Serializers, scope=injector.singleton)
        binder.bind(DeSerializers, DeSerializers, scope=injector.singleton)



    @inject_context_di()
    def retrieve_bind_props(self, to_register: typing.Type[T], ctx: typing.Optional[InjectionContextInjector]):
        if not ctx.contains_binding(to_register):
            ctx.register_config_properties(to_register, to_register.fallback)
