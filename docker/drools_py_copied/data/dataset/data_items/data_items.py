import abc
from typing import Generic, TypeVar

U = TypeVar("U")
DataTypeT = TypeVar("DataTypeT")


class DataItem(Generic[DataTypeT], abc.ABC):

    def __init__(self, src: DataTypeT, tgt: DataTypeT):
        self._tgt = tgt
        self._src = src

    @property
    def src(self) -> DataTypeT:
        return self._src

    @property
    def tgt(self) -> DataTypeT:
        return self._tgt


class ValidationItem(Generic[U], DataItem[U]):
    def __init__(self, src: DataTypeT, tgt: DataTypeT):
        super().__init__(src, tgt)


class TestItem(Generic[U], DataItem[U]):
    def __init__(self, src: DataTypeT, tgt: DataTypeT):
        super().__init__(src, tgt)


class TrainingItem(Generic[U], DataItem[U]):
    def __init__(self, src: DataTypeT, tgt: DataTypeT):
        super().__init__(src, tgt)


class TextTrainingItem(TrainingItem[str]):

    def __init__(self, src: str, tgt: str):
        super().__init__(src, tgt)
