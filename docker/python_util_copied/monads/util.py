import itertools
from typing import TypeVar, Iterable, Optional, Callable

T = TypeVar("T")
G = TypeVar("G")


def flatten_iterable(iterable_val: Iterable[Iterable[T]]) -> Iterable[T]:
    return [out_item for out_item
            in iterable_val
            for out_item in out_item]


def flatmap_opt(func: Callable[[T], G], iterable_val: Optional[T]) -> Optional[G]:
    if iterable_val:
        return func(iterable_val)
    return None


def flatmap(func: Callable[[T], G], *iterable) -> G:
    return itertools.chain.from_iterable(map(func, *iterable))


def add_modality_key(modality_key, params):
    if modality_key not in params.keys():
        params[modality_key] = []
