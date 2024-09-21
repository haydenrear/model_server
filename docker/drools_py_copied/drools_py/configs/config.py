import abc
import copy
import enum
import inspect
import logging
import typing
from typing import get_origin, get_args

import injector

import python_di
from drools_py.serialize.to_from_json import ToFromJsonDict
from python_di.inject.context_builder.inject_ctx import inject_context_di
from python_di.inject.context_builder.injection_context import InjectionContext
from python_di.inject.injector_provider import InjectionContextInjector
from python_di.inject.profile_composite_injector.scopes.prototype_scope import prototype_scope_decorator_factory
from python_di.inject.profile_composite_injector.composite_injector import profile_scope
from python_util.logger.logger import LoggerFacade
from python_util.reflection.reflection_utils import get_all_fn_param_types, is_ignore_param


class ConfigType(enum.Enum):
    Test = "test"
    Validation = "validation"
    MainProfile = "main_profile"
    TestBatches = "test_batches"

    @classmethod
    def from_value(cls, v: str):
        if v == 'test':
            return ConfigType.Test
        elif v == 'test_batches':
            return ConfigType.TestBatches
        elif v == 'validation':
            return ConfigType.Validation
        else:
            return ConfigType.MainProfile


T = typing.TypeVar("T")


class ConfigOverrides:

    @property
    def override_map(self):
        return self._override_map

    @override_map.setter
    def override_map(self, override_map):
        self._override_map = override_map

    # @staticmethod
    # @abc.abstractmethod
    # def prod_properties(**kwargs) -> dict:
    #     pass

    @classmethod
    def delete_extra_params(cls, to_override, constructor_params, skip_params: list[str] = None):
        if not to_override:
            return to_override
        to_delete = []
        if to_override and len(to_override) != 0:
            to_override = copy.copy(to_override)
        for to_override_key, _ in to_override.items():
            if (to_override_key not in constructor_params
                    and not is_ignore_param(to_override_key)
                    and to_override_key in to_override.keys()):
                to_delete.append(to_override_key)
        if len(to_override) != 0:
            to_override = copy.copy(to_override)
        for key in to_delete:
            if not skip_params or key not in skip_params:
                del to_override[key]

        return to_override

    @classmethod
    def update_override_no_deletion(cls, to_override, starting):
        if starting is None:
            return to_override
        for key, val in to_override.items():
            starting[key] = val

        return starting

    @classmethod
    def update_override(cls, to_override, starting, skip_delete_params: list[str] = None):
        constructor_params = get_all_fn_param_types(cls.__init__).keys()
        to_override = cls.delete_extra_params(to_override, constructor_params, skip_delete_params)
        starting = cls.delete_extra_params(starting, constructor_params, skip_delete_params)
        if starting is None:
            return to_override
        for key, val in to_override.items():
            if val is not None:
                starting[key] = val

        return starting

    @classmethod
    def delegate_config_creation(cls, config_type: ConfigType, profile: str):
        return cls.get_bean(config_type, cls, scope=prototype_scope_decorator_factory(profile)())

    @classmethod
    @inject_context_di()
    def get_bean(cls, config_type: ConfigType, ty: typing.Type[T],
                 scope: injector.ScopeDecorator = profile_scope,
                 ctx: typing.Optional[InjectionContextInjector] = None,
                 **kwargs) -> typing.Optional[T]:
        found = ctx.get_interface(ty, profile=config_type.name, scope=scope, **kwargs)
        if found is not None:
            return found
        else:
            LoggerFacade.error(f"Failed to retrieve bean {ty} from {cls} for profile {config_type.name}")


class Config(ToFromJsonDict, ConfigOverrides, abc.ABC):

    def update_arg(self, key: str, value):
        if hasattr(self, key):
            setattr(self, key, value)

        for _, val in vars(self).items():
            if isinstance(val, Config) and hasattr(val, key):
                val.update_arg(key, value)

    @classmethod
    def copy_add_props(cls, input_dict: dict, to_add: dict):
        copied = copy.copy(input_dict)
        for key, val in to_add.items():
            copied[key] = val

        return copied

    @classmethod
    def get_from_kwarg_or_default(cls, arg: dict, key, default_value):
        if key in arg.keys():
            return arg[key]
        return default_value

    @classmethod
    def get_prototype_bean(cls, config_type: ConfigType, **kwargs) -> T:
        """
        Retrieve prototype bean with dependencies as determined by profile in ConfigType. All kwargs provided
        will override factory args.
        :param config_type:
        :param kwargs:
        :return:
        """
        return cls.get_build_props(config_type, **kwargs)

    @classmethod
    def build_config_type_config(cls, config_type: ConfigType, **kwargs) -> T:
        return cls.get_build_props(config_type, **kwargs)

    @classmethod
    def get_properties(cls, config_type: ConfigType, **kwargs) -> dict:
        return cls.get_build_props(config_type).to_self_dictionary()

    @classmethod
    def validation_properties(cls, **kwargs) -> dict:
        return cls.get_properties(ConfigType.Validation, **kwargs)

    @classmethod
    def test_properties(cls, **kwargs) -> dict:
        return cls.get_properties(ConfigType.Test, **kwargs)

    @classmethod
    def build_test_config(cls, **kwargs):
        return cls.build_config_type(ConfigType.Test, **kwargs)

    @classmethod
    def build_validation_config(cls, **kwargs):
        return cls.build_config_type(ConfigType.Validation, **kwargs)

    @classmethod
    def get_build_props(cls, config_type: ConfigType, **kwargs):
        decorator_factory = prototype_scope_decorator_factory(config_type.name)
        decorator = decorator_factory()
        return cls.get_bean(
            config_type,
            cls,
            decorator,
            **kwargs
        )

    @classmethod
    def props_minus_dyn(cls, config_type: ConfigType, **kwargs):
        """
        This is a function that looks back into the previous frame and retrieves the arguments passed in and then
        removes them from the properties. The properties methods return the properties for a particular profile.
        But then they return all properties required in the function params. So then if you provide your own to
        override, it throws an error because there are extra properties defined. So find which ones and don't include
        them. IMPORTANT - MUST USE KWARG IN ORDER FOR IT TO REMOVE THE ARG from the test props - or else you'll get
        a duplicate kwarg exception.
        :param config_type:
        :param kwargs:
        :return:
        """
        consts = inspect.currentframe().f_back.f_code.co_consts[1]
        removed = cls.props_minus(config_type, [i for i in consts], **kwargs)
        return removed

    @classmethod
    def props_minus(cls, config_type: ConfigType, minus_keys: list[str], **kwargs):
        config_ty = cls.build_config_type_config(config_type, **kwargs)
        out = config_ty.to_self_dictionary()
        for key in filter(lambda k: k in out.keys(), minus_keys):
            del out[key]
        return out

    @classmethod
    def build_config_type(cls, config_type: ConfigType, **kwargs) -> T:
        return cls.build_config_type_config(config_type, **kwargs)

