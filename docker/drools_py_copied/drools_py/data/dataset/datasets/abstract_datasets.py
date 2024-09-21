import abc
import enum
import typing
from typing import Generic, Optional, TypeVar

from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.data.dataset.data_items.data_items import DataItem
from drools_py.data.dataset.dataset_decorator.get_item_decorator import GetItemDecorator
from drools_py.serialize.serializable_types import SerializableEnum

DataItemT = TypeVar("DataItemT")
DataT = typing.TypeVar("DataT", covariant=True, bound=DataItem)


class DatasetType(SerializableEnum):
    Train = 0
    Test = 1
    Validation = 2


class DataType(SerializableEnum):
    Image = 0
    Text = 1


class DatasetSourceType(SerializableEnum):
    Huggingface = enum.auto()


class AbstractLoadStrategy(abc.ABC):
    pass


class AbstractDataItemIter(abc.ABC, Generic[DataItemT]):
    pass


class AbstractDataset(Dataset[DataT], abc.ABC, Generic[DataT]):

    @property
    @abc.abstractmethod
    def decorated(self):
        pass

    def to_data_loader(self) -> DataLoader:
        return DataLoader(self)

    def next_item(self) -> Optional[DataT]:
        return self.item()

    def item(self, idx: Optional[int] = None) -> Optional[DataT]:
        return self.decorated[idx]

    def __getitem__(self, item):
        return self.item(item)


class DataIterDecoratedFactoryConfig(Config):
    def __init__(self, data_type: DatasetSourceType,
                 dataset_type: DatasetType):
        self.dataset_type = dataset_type
        self.data_type = data_type

