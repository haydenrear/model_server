import abc

from python_util.collections.topological_sort import HasDependencies


class GetItemDecorator(HasDependencies[type], abc.ABC):
    @abc.abstractmethod
    def get_item(self, idx) -> ...:
        pass

    def has_item(self) -> bool:
        return False

    @abc.abstractmethod
    def get_dependencies(self) -> list[type]:
        pass

    def self_id(self) -> type:
        return type(self)
