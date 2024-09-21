import abc


class Ordered(abc.ABC):
    @abc.abstractmethod
    def order(self) -> int:
        pass