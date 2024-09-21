import importlib
import inspect
import logging
import typing
from typing import Any


def class_name_str(value: Any):
    return f'{str(value.__class__.__name__)}'


def get_module_name_str(value: Any):
    try:
        return str(value.__module__)
    except:
        return None


def get_class_obj_from_module(class_name, module):
    try:
        module = importlib.import_module(module)
        class_obj = getattr(module, class_name)
        return class_obj
    except Exception as e:
        return None


def is_subclass_of(class_name: str, to_test: type, module: str):
    class_obj = get_class_obj_from_module(class_name, module)
    if class_obj and to_test:
        try:
            is_subclass = issubclass(class_obj, to_test)
            return is_subclass
        except Exception as e:
            logging.error(f"Error checking subclass: {class_obj} and {to_test}: {e}")
    return False


def get_fn_param_types(fn) -> dict:
    return {v.name: (v.annotation, v.default) for k, v in inspect.signature(fn).parameters.items()
            if not is_empty_inspect(v.annotation)}


def get_all_fn_param_types(fn) -> dict:
    return {v.name: (v.annotation, v.default) for k, v in inspect.signature(fn).parameters.items()}


def get_all_fn_param_types_no_default(fn) -> dict:
    return {v.name: v.annotation for k, v in inspect.signature(fn).parameters.items()}


def is_type_instance_of(type_to_compare_to: type, type_to_compare: type) -> bool:
    if type_to_compare is type_to_compare_to:
        return True
    if hasattr(type_to_compare_to, "__origin__"):
        does_origin_compare = type_to_compare_to.__origin__ is type_to_compare
        if does_origin_compare:
            return True
    if hasattr(type_to_compare, "__origin__"):
        does_origin_compare = type_to_compare.__origin__ is type_to_compare_to
        if does_origin_compare:
            return True
    try:
        is_instance = isinstance(type_to_compare_to, type_to_compare) or isinstance(type_to_compare, type_to_compare_to)
    except:
        is_instance = False

    if is_instance:
        return is_instance

    if hasattr(type_to_compare, '__bases__'):
        for t in type_to_compare.__bases__:
            if t == type_to_compare_to:
                return True
    if hasattr(type_to_compare_to, '__bases__'):
        for t in type_to_compare_to.__bases__:
            if t == type_to_compare:
                return True

    return False


def is_ignore_param(param_name) -> bool:
    return param_name == "self" or param_name == "kwargs" or param_name == "args"


def get_return_type(fn) -> typing.Type:
    return inspect.signature(fn).return_annotation


def empty_inspect_ex():
    pass


p = inspect.signature(empty_inspect_ex)


def is_empty_inspect(val) -> bool:
    return val == p.empty


def is_optional_ty(ty_value) -> bool:
    return hasattr(ty_value, '_name') and 'Optional' == ty_value._name

