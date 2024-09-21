import typing

import injector
from datasets import DatasetDict
from torch.utils.data import Dataset, Dataset as TorchDataset

from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_factory import ConfigFactory
from drools_py.data.constant import DECORATED_DATA_METADATA
from drools_py.data.dataset.dataset_decorator.get_item_decorator import GetItemDecorator
from drools_py.data.dataset.dataset_decorator.split_decorator import DecoratedDataMetadata
from drools_py.data.dataset.datasets.abstract_datasets import DatasetSourceType, DatasetType
from python_util.collections.topological_sort import topological_sort
from python_di.configs.component import component
from python_di.configs.prototype import prototype_factory, prototype_scope_bean
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_util.logger.logger import LoggerFacade


@prototype_scope_bean()
class DecoratedDatasetConfig(Config):
    @prototype_factory()
    def __init__(self,
                 underlying: Dataset,
                 dataset_type: DatasetSourceType,
                 data_type: DatasetType,
                 get_item_decorators: typing.List[GetItemDecorator]):
        self.data_type = data_type
        self.dataset_type = dataset_type
        self.get_item_decorators = get_item_decorators
        self.underlying = underlying


class DecoratedDataset(Dataset):

    def __init__(self, decorated_config: DecoratedDatasetConfig):
        self.get_item_decorators = [i for i in reversed(topological_sort(decorated_config.get_item_decorators))]
        self.data = decorated_config.underlying
        self.idx = 0

    def load(self, idx: typing.Optional[int] = None):
        if idx is not None:
            self.idx = idx
        else:
            self.idx += 1
        return self.data[self.idx]

    def __len__(self):
        if hasattr(self.data, '__len__'):
            return len(self.data)
        else:
            LoggerFacade.warn(f"Dataset {self.data} did not contain __len__ and was not Sized. Returning 100000 for "
                              f"pytorch sampling")
            return 100000

    def reset(self):
        self.idx = 0

    def __getitems__(self, possible_batched_index):
        return [self.__getitem__(max(possible_batched_index))]


    def __getitem__(self, idx):
        free_item = None
        for i, d in enumerate(self.get_item_decorators):
            if d.has_item():
                free_item = d.get_item()
                continue

            if free_item is not None:
                free_item = d.get_item(free_item)

        if free_item is not None:
            return self._update_decorated(free_item)

        i = None
        d = self.load(idx)
        while i is None and d is not None:
            i = d
            for decorator in self.get_item_decorators:
                i = decorator.get_item(i)
            if i is not None:
                break
            d = self.load(idx)

        return self._update_decorated(i)

    @staticmethod
    def _update_decorated(free_item):
        if DECORATED_DATA_METADATA in free_item.keys():
            decorated_data: DecoratedDataMetadata = free_item[DECORATED_DATA_METADATA]
            for k, v in decorated_data.to_dict().items():
                free_item[k] = v
            del free_item[DECORATED_DATA_METADATA]
        return free_item


class DecoratedDatasetConfigFactory(ConfigFactory):
    def __init__(self):
        super().__init__()

    def create(self, config: DecoratedDatasetConfig, **kwargs):
        if config.dataset_type == DatasetSourceType.Huggingface:
            if isinstance(config.underlying, DatasetDict):
                return DecoratedDataset(
                    DecoratedDatasetConfig(config.underlying[config.data_type.name.lower()],
                                           config.dataset_type,
                                           config.data_type, config.get_item_decorators))
            else:
                assert isinstance(config.underlying, Dataset)
                return DecoratedDataset(config)

        raise ValueError(f"{config.dataset_type} was not a known dataset type.")
