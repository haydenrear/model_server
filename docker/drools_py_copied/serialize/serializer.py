import abc
import enum
import importlib
import json
import os
import typing
from typing import TypeVar

import injector
import torch.nn
from sqlalchemy import String

from drools_py.configs.config import Config
from drools_py.serialize.base_serializer import BaseSerializer, DeSerializers, Serializers
from drools_py.serialize.serializer_properties import SerializerProperties
from python_util.reflection.reflection_utils import (get_all_fn_param_types, class_name_str, get_module_name_str)
from python_util.io_utils.file_dirs import create_random_file_name
from drools_py.serialize.serializable_types import NoArgsSerializable, SerializableEnum
from drools_py.serialize.to_from_json import ToFromJsonDict
from drools_py.configs.config_models import ConfigOption

T = TypeVar("T")

U = TypeVar("U")

ToSerializeT = TypeVar("ToSerializeT")




class SerializerFields(enum.Enum):
    ValueType = "value_type"
    Clzz = "clzz"
    Module = "module"
    ConfigValue = "config_value"
    ConfigType = "config_type"
    ModuleClzz = "module_clzz"
    ModuleId = "module_id"
    ConfigOption = "config_option"


class SerializerType(enum.Enum):
    TorchModule = enum.auto()
    Config = enum.auto()
    ConfigOption = enum.auto()
    Other = enum.auto()
    Dict = enum.auto()
    SerializableEnum = enum.auto()
    Tuple = enum.auto()
    NoArgsCreator = enum.auto()
    Reflecting = enum.auto()
    SqlAlchemyStr = enum.auto()
    Set = enum.auto()
    MetadataProvider = enum.auto()
    List = enum.auto()
    Optional = enum.auto()
    TorchTensor = enum.auto()


class Serializer(BaseSerializer, typing.Generic[ToSerializeT]):

    @injector.inject
    def __init__(self,
                 serializer_properties: SerializerProperties,
                 serializers: Serializers = None,
                 deserializers: DeSerializers = None):
        self.serializer_properties = serializer_properties
        if serializers is not None:
            self.serializers: typing.List[Serializer] \
                = typing.cast(typing.List[Serializer], serializers.base_serializers)
        if deserializers is not None:
            self.de_serializers: typing.List[Serializer] \
                = typing.cast(typing.List[Serializer], serializers.base_serializers)
            self.dict_serializer = deserializers.dict_ser()
            self.primitive_serializer = deserializers.primitive_ser()

    def serialize(self, value: ToSerializeT, current: typing.Optional[dict] = None) -> dict:
        if current and len(current) != 0:
            return current
        return {
            "clzz": class_name_str(value),
            "module": get_module_name_str(value),
            "config_value": None,
            "config_type": None,
            "module_clzz": None,
            "module_id": None,
            "value_type": []
        }

    def matches_de_serializer_type(self) -> list[SerializerType]:
        pass

    def matches_types(self) -> list[type]:
        pass

    def de_serialize_args(self, dict_to_deserialize: dict) -> typing.Optional[dict]:
        return None

    def de_serialize_args_base(self, dict_to_deserialize: dict, constructor) -> dict:

        out_args = {}

        for key, val in dict_to_deserialize.items():
            if not val:
                out_args[key] = None
            try:
                if isinstance(key, str) and key.startswith('_'):
                    key = key.replace('_', '', 1)
                out_args[key] = self.delegate_de_serialize(val)
            except:
                out_args[key] = self.delegate_de_serialize(val)

        fn_params = get_all_fn_param_types(constructor)
        to_delete = []

        for de_ser in self.get_de_serializer(dict_to_deserialize):
            args = de_ser.de_serialize_args(dict_to_deserialize)
            if not args:
                continue
            for key, val in args.items():
                out_args[key] = val

        for key, val in out_args.items():
            if key not in fn_params.keys():
                to_delete.append(key)

        for delete in to_delete:
            del out_args[delete]

        return out_args

    def de_serialize(self, dict_to_deserialize: dict) -> ToSerializeT:
        out_args, type_to_create = self.get_deserialize_items(dict_to_deserialize)
        return type_to_create(**out_args)

    def get_constructor(self, dict_to_deserialize: dict):
        return None

    def get_constructor_base(self, dict_to_deserialize: dict):
        if (SerializerFields.Module.value in dict_to_deserialize.keys()
                and SerializerFields.Clzz.value in dict_to_deserialize.keys()):
            mod = dict_to_deserialize[SerializerFields.Module.value]
            next_one = dict_to_deserialize[SerializerFields.Clzz.value]
            out = importlib.import_module(mod, next_one)
            constructor = out.__dict__[next_one]
            return constructor

    def get_deserialize_items(self, dict_to_deserialize) -> (dict, type):
        constructor = self.get_constructor(dict_to_deserialize)

        del dict_to_deserialize[SerializerFields.Module.value]
        del dict_to_deserialize[SerializerFields.Clzz.value]

        return self.de_serialize_args_base(dict_to_deserialize, constructor), constructor

    def delegate_de_serialize(self, val):
        if self.primitive_serializer.is_primitive(val):
            return val
        elif isinstance(val, dict) and SerializerFields.ValueType.value not in val.keys():
            return val
        elif (isinstance(val, list) or isinstance(val, typing.Iterable)) and not isinstance(val, dict):
            out_val = type(val)([i for i in val])
            return out_val
        elif isinstance(val, dict) and val[SerializerFields.Clzz.value] == 'dict':
            return self.dict_serializer.de_serialize_args(val)
        elif not isinstance(val, dict):
            return val

        de_serializer = self.get_de_serializer(val)

        for d in de_serializer:
            if isinstance(d, ConstructableSerializer):
                return d.construct(val)

        constructor = None
        for de_ser in de_serializer:
            constructor = de_ser.get_constructor(val)

        if not constructor:
            constructor = self.get_constructor_base(val)

        if not constructor:
            return val

        args = self.de_serialize_args_base(val, constructor)
        return constructor(**args)

    def matches_serialize(self, value: ToSerializeT) -> bool:
        return any([isinstance(value, ty) for ty in self.matches_types()])

    def matches_deserialize(self, value) -> bool:
        serializer_type = self.matches_de_serializer_type()
        return any([ty.value in value[SerializerFields.ValueType.value] for ty in serializer_type])

    def delegate_serialize(self, value: ToSerializeT, curr=None):
        serializer = self.get_serializer(value)
        current = {} if not curr else curr
        if len(serializer) != 0:
            for ser in serializer:
                current = ser.serialize(value, current)
            return self.delete_none_types(current)
        else:
            if isinstance(value, typing.Iterable) and not isinstance(value, str) and not isinstance(value,
                                                                                                    torch.Tensor):
                return [
                    self.delete_none_types(self.delegate_serialize(val)) for val in value
                ]
            else:
                return self.delete_none_types(self.serialize(value, current))

    def delete_none_types(self, to_delete_from):
        to_delete = []
        for key, val in to_delete_from.items():
            if not val and key in SerializerFields.__members__.values():
                to_delete.append(key)

        for delete in to_delete:
            del to_delete_from[delete]

        return to_delete_from

    def get_serializer(self, value: ToSerializeT):
        serializers = []
        for serializer in self.serializers:
            if serializer.matches_serialize(value=value):
                serializers.append(serializer)

        return serializers

    def does_match(self, value, serializer):
        return serializer.matches_serialize(value)

    def get_de_serializer(self, value: dict):
        serializers = []
        for serializer in self.de_serializers:
            if serializer.matches_deserialize(value):
                serializers.append(serializer)

        return serializers


class ConstructableSerializer(Serializer[ToSerializeT],
                              typing.Generic[ToSerializeT],
                              abc.ABC):
    @abc.abstractmethod
    def construct(self, input_dict: dict) -> typing.Optional[SerializerType]:
        pass


class PrimitiveSerializer(Serializer):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Other]

    def is_primitive(self, val):
        return any([isinstance(val, i) for i in self.matches_types()])

    def matches_types(self) -> list[type]:
        return [(int, str, bool, float)]

    def serialize(self, value: ToSerializeT, current=None) -> dict:
        starting = super().serialize(value, current)
        starting[SerializerFields.ValueType.value].append(SerializerType.Other.value)
        starting["primitive_config_value"] = value
        return starting

    def get_constructor(self, dict_to_deserialize: dict):
        return lambda primitive_constructor: primitive_constructor

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        if SerializerFields.Clzz.value not in dict_to_deserialize.keys() \
                or "primitive_config_value" not in dict_to_deserialize.keys() \
                or dict_to_deserialize[SerializerFields.Clzz.value] == 'NoneType':
            return {"primitive_constructor": None}
        try:
            primitive_value = eval(dict_to_deserialize['clzz'])(dict_to_deserialize["primitive_config_value"])
            return {"primitive_constructor": primitive_value}
        except Exception as e:
            raise ValueError(f"Failure to create primitive value from {dict_to_deserialize} clzz type: "
                             f"{dict_to_deserialize['clzz']} with error {e}.")


class SqlAlchemyStringSerializer(Serializer[String]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.SqlAlchemyStr]

    def matches_types(self) -> list[type]:
        return [String, typing.Type]

    def serialize(self, value: ToSerializeT, current=None) -> dict:
        starting = super().serialize(value, current)
        starting[SerializerFields.ValueType.value] = [SerializerType.SqlAlchemyStr.value]
        starting["sql_alchemy_config_value"] = str(value)
        return starting

    def get_constructor(self, dict_to_deserialize):
        return lambda sql_alchemy_config_value: String(len(sql_alchemy_config_value), sql_alchemy_config_value)

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {"sql_alchemy_config_value": dict_to_deserialize["sql_alchemy_config_value"]}


class NoArgsTorchModuleSerializer(Serializer[torch.nn.Module]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.TorchModule]

    def matches_types(self) -> list[type]:
        return [torch.nn.Module]

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {}

    def serialize(self, value: ToSerializeT, current=None) -> dict:
        starting = super().serialize(value, current)
        starting["module_clzz"] = str(type(value))
        starting["module_id"] = str(id(value))
        starting[SerializerFields.ValueType.value].append(SerializerType.TorchModule.value)
        return starting


class ConfigOptionSerializer(Serializer[ConfigOption]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.ConfigOption]

    def matches_types(self) -> list[type]:
        return [ConfigOption]

    def serialize(self, value: ToSerializeT, current = None):
        starting = super().serialize(value, current)
        starting[SerializerFields.ConfigOption.value] = super().delegate_serialize(value.config_option)
        starting[SerializerFields.ConfigType.value] = str(type(value))
        starting[SerializerFields.ValueType.value].append(SerializerType.ConfigOption.value)
        return starting

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {SerializerFields.ConfigOption.value: dict_to_deserialize[SerializerFields.ConfigOption.value]}


class DictSerializer(Serializer[dict]):
    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Dict]

    def matches_types(self) -> list[type]:
        return [dict]

    def serialize(self, value: ToSerializeT, curr=None):
        if curr:
            out_values = curr
        else:
            out_values = {}
        if SerializerFields.ValueType.value not in out_values.keys():
            out_values[SerializerFields.ValueType.value] = [SerializerType.Dict.value]
        else:
            out_values[SerializerFields.ValueType.value].append(SerializerType.Dict.value)
        for key, val in value.items():
            serialized = super().delegate_serialize(val)
            out_values[key] = serialized
        out_values[SerializerFields.Clzz.value] = "dict"
        return out_values

    def get_constructor(self, dict_to_deserialize: dict):
        return lambda x: x

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        del dict_to_deserialize[SerializerFields.Clzz.value]
        del dict_to_deserialize[SerializerFields.ValueType.value]

        new_out = {}
        for key, val in dict_to_deserialize.items():
            new_out[key] = super().delegate_de_serialize(val)
        return new_out


class TupleSerializer(Serializer[typing.Tuple]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Tuple]

    def matches_types(self) -> list[type]:
        return [tuple, typing.Tuple]

    def serialize(self, value: ToSerializeT, curr=None):
        out_values = super().serialize(value, curr)
        out_values["tuple_value"] = tuple([super().delegate_serialize(val) for val in value])
        out_values[SerializerFields.Clzz.value] = "tuple"
        out_values[SerializerFields.ValueType.value].append(SerializerType.Tuple.value)
        return out_values

    def get_constructor(self, dict_to_deserialize: dict):
        return lambda tuple_constructor: tuple([i for i in tuple_constructor])

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        del dict_to_deserialize[SerializerFields.Clzz.value]
        del dict_to_deserialize[SerializerFields.ValueType.value]
        return {
            "tuple_constructor": [super().delegate_de_serialize(val) for val in dict_to_deserialize["tuple_value"]]}


class SetSerializer(Serializer[typing.Set]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def get_constructor(self, dict_to_deserialize: dict):
        return self.set_constructor_constructor

    def set_constructor_constructor(self, set_constructor):
        return set([i for i in set_constructor])

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Set]

    def matches_types(self) -> list[type]:
        return [set, typing.Set]

    def serialize(self, value: ToSerializeT, curr=None):
        out_values = super().serialize(value, curr)
        out_values["set_value"] = [super().delegate_serialize(val) for val in value]
        out_values[SerializerFields.Clzz.value] = "set"
        out_values[SerializerFields.ValueType.value].append(SerializerType.Set.value)
        return out_values

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {"set_constructor": [super().delegate_de_serialize(val)
                                    for val in dict_to_deserialize["set_value"]]}


class ListSerializer(Serializer[typing.List]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def get_constructor(self, dict_to_deserialize: dict):
        return lambda list_constructor: list([i for i in list_constructor])

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.List]

    def matches_types(self) -> list[type]:
        return [list, typing.List]

    def serialize(self, value: ToSerializeT, curr=None):
        out_values = super().serialize(value, curr)
        out_values["list_value"] = [super().delegate_serialize(val) for val in value]
        out_values[SerializerFields.Clzz.value] = "list"
        out_values[SerializerFields.ValueType.value].append(SerializerType.List.value)
        return out_values

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {"list_constructor": [super().delegate_de_serialize(val)
                                     for val in dict_to_deserialize["list_value"]]}


class SerializableEnumSerializer(Serializer[SerializableEnum]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.SerializableEnum]

    def matches_types(self) -> list[type]:
        return [SerializableEnum]

    def de_serialize_args(self, dict_to_deserialize) -> ToSerializeT:
        return {"value": dict_to_deserialize}

    def get_constructor(self, dict_to_deserialize: dict):
        return self.create_enum

    def create_enum(self, value):
        base = super().get_constructor_base(value)
        out_enum = getattr(base, value[SerializerFields.ConfigValue.value])
        return out_enum

    def serialize(self, value: SerializableEnum, curr=None) -> dict:
        out_dir = super().serialize(value, curr)
        out_dir[SerializerFields.ConfigValue.value] = value.name
        out_dir[SerializerFields.ValueType.value].append(SerializerType.SerializableEnum.value)
        return out_dir


class ReflectingSerializable(ToFromJsonDict, abc.ABC):
    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)


class ReflectingSerializer(Serializer[ReflectingSerializable]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Reflecting]

    def matches_types(self) -> list[type]:
        return [ReflectingSerializable]

    def serialize(self, value: ToSerializeT, curr=None) -> dict:
        starting = super().serialize(value, curr)
        starting[SerializerFields.ValueType.value].append(SerializerType.Reflecting.value)
        for key, val in value.__dict__.items():
            starting[key] = super().delegate_serialize(val)
        return starting


class NoArgsCreatorSerialize(Serializer[NoArgsSerializable]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.NoArgsCreator]

    def matches_types(self) -> list[type]:
        return [NoArgsSerializable]

    def serialize(self, value: ToSerializeT, curr=None) -> dict:
        out_dir = super().serialize(value, curr)
        out_dir[SerializerFields.Clzz.value] = class_name_str(value)
        out_dir[SerializerFields.ValueType.value].append(SerializerType.NoArgsCreator.value)
        return out_dir

    def de_serialize_args(self, dict_to_deserialize: dict) -> dict:
        return {}


class ConfigSerializer(Serializer[Config]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.Config]

    def matches_types(self) -> list[type]:
        return [Config]

    def get_constructor(self, dict_to_deserialize: dict):
        return self.get_constructor_base(dict_to_deserialize)

    def serialize(self, value: ToSerializeT, curr=None) -> dict:
        out_dir = super().serialize(value, curr)
        out_dir[SerializerFields.ValueType.value].append(SerializerType.Config.value)

        d = vars(value)

        for key, val in d.items():
            if key in get_all_fn_param_types(value.__init__).keys():
                out_dir[key] = super().delegate_serialize(val)

        return out_dir


class TorchTensorSerializer(ConstructableSerializer[torch.Tensor]):

    @injector.inject
    def __init__(self, serializer_properties: SerializerProperties):
        super().__init__(serializer_properties)

    def construct(self, input_dict: dict) -> typing.Optional[SerializerType]:
        if 'path' in input_dict.keys():
            return torch.load(input_dict['path'])

    def matches_de_serializer_type(self) -> list[SerializerType]:
        return [SerializerType.TorchTensor]

    def matches_types(self) -> list[type]:
        return [torch.Tensor]

    def get_constructor(self, dict_to_deserialize: dict):
        return self.get_constructor_base(dict_to_deserialize)

    def serialize(self, value: ToSerializeT, curr=None) -> dict:
        out_serialized = super().serialize(value, curr)
        out_serialized[SerializerFields.ValueType.value].append(SerializerType.TorchTensor.name)
        out_directory = os.path.join(self.serializer_properties.torch_serialize_base_out_dir,
                                     create_random_file_name('tch'))
        torch.save(value, out_directory)
        out_serialized["path"] = out_directory
        return out_serialized

    def matches_deserialize(self, value) -> bool:
        return (SerializerFields.ValueType.value in value.keys()
                and SerializerType.TorchTensor.name in value[SerializerFields.ValueType.value])
