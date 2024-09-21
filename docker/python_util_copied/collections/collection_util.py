import collections
import typing
from typing import TypeVar

T = typing.TypeVar("T")


def retrieve_all_indices_of(from_collection: list[T], to_find_idx_for: T) -> list[int]:
    found = from_collection.index(to_find_idx_for)
    out = [found]
    found += 1
    while found is not None and found < len(from_collection):
        try:
            found = from_collection.index(to_find_idx_for, found)
            out.append(found)
            found += 1
        except ValueError:
            break

    return out


def first_key(input: dict, sorted_by_key=True):
    if not input or len(input) == 0:
        return None
    if sorted_by_key:
        return next(iter(sorted(input.keys())))
    else:
        return next(iter(input.keys()))


def collect_multimap(input_collection, partition_fn):
    out_map = {}
    for i in input_collection:

        f = partition_fn(i)
        if f not in out_map.keys():
            if isinstance(i, typing.List | list | set):
                out_map[f] = i
            else:
                out_map[f] = [i]
        else:
            if isinstance(i, typing.List | list | set):
                out_map[f].extend(i)
            else:
                out_map[f].append(i)

    return out_map


def merge_multivalue_maps(to_add: dict, starting_map: dict):
    for k, v in to_add.items():
        if k not in starting_map.keys():
            starting_map[k] = v
        else:
            starting_map[k].extend(v)


def last_two(size: int, reversed_key_iter):
    last = None
    next_to_last = None
    if size > 1:
        last = next(reversed_key_iter)
        next_to_last = next(reversed_key_iter)
    elif size == 1:
        last = next(reversed_key_iter)
    return last, next_to_last


def first(input: dict):
    key = first_key(input)
    if key:
        return input[key]


T = TypeVar("T")


def first_from_iter(input: list[T]):
    if len(input) == 0:
        return None
    return next(iter(input))
