import copy


def copy_with_override(value, **kwargs):
    new_value = copy.copy(value)
    for key, val in kwargs.items():
        setattr(new_value, key, val)
    return new_value

