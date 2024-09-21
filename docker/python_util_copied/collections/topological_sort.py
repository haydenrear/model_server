import abc
import typing

T = typing.TypeVar("T")


class HasDependencies(abc.ABC, typing.Generic[T]):
    @abc.abstractmethod
    def get_dependencies(self) -> list[T]:
        pass

    @abc.abstractmethod
    def self_id(self) -> T:
        pass


def topological_sort(objects: list[HasDependencies]):
    def dfs(obj, visited, order):
        if obj in visited:
            return
        visited.add(obj)
        for dependency in obj.get_dependencies():
            dependency_obj = obj_map.get(dependency)
            if dependency_obj in visited:
                # are we in recursion of this item?
                if dependency_obj not in order:
                    raise ValueError(f"Circular dependency detected for {obj.self_id()} when parsing {dependency}! "
                                     f"Topological sort deemed a failure.")
            else:
                dfs(dependency_obj, visited, order)
        order.append(obj)

    visited = set()
    order = []

    obj_map = {obj.self_id(): obj for obj in objects}

    for obj in objects:
        dfs(obj, visited, order)

    return list(reversed(order))
