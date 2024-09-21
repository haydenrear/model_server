import abc
import typing

from drools_py.configs.config_factory import ConfigFactory
from drools_py.data.dataset.data_items.data_items import TextTrainingItem
from drools_py.data.dataset.datasets.abstract_datasets import DataType, DataT


class DataItemFactory(typing.Generic[DataT], abc.ABC):
    @abc.abstractmethod
    def produce(self, input_value) -> DataT:
        pass


class TextTrainingItemFactory(DataItemFactory[TextTrainingItem]):
    def produce(self, input_value) -> DataT:
        return TextTrainingItem(input_value[0], input_value[1])


class DataItemConfigFactory(ConfigFactory):
    def create(self, dataset_type: DataType, **kwargs):
        if dataset_type == DataType.Text:
            return TextTrainingItemFactory()
        else:
            raise ValueError(f"{dataset_type} is not implemented as a datatype yet.")
