from typing import Any, List

import torch
from pytorch_lightning import LightningDataModule
from typing_extensions import Self

from lightning.pytorch.utilities import CombinedLoader
from torch.utils.data import DataLoader

from drools_py.data.constant import ATTENTION_MASK, INPUT_IDS, N, OUT_OF
from drools_py.data.dataset.dataset_config import DatasetConfig
from drools_py.data.dataset.dataset_decorator.decorated_dataset import DecoratedDatasetConfig, DecoratedDataset
from drools_py.data.dataset.datasets.abstract_datasets import AbstractDataset, DataT
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.pytorch_util import create_torch_size_log


class TorchDatasetDecorator(AbstractDataset[DataT], LightningDataModule):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__()
        self.dataset_config = dataset_config
        downloaded_dataset = self.dataset_config.load_config.create(dataset_config)
        self._decorated = dataset_config.decorated.create(
            DecoratedDatasetConfig.get_prototype_bean(dataset_config.config_type,
                                                      underlying=downloaded_dataset)
        )

    def reset(self):
        self._decorated.reset()


    def to_data_loader(self) -> DataLoader:
        """
        Loads data as dict:
        INPUT_IDS: [batch_size, seq_len, dim]
        ATTENTION_MASK: [batch_size, seq_len]
        :return:
        """
        return DataLoader(self._decorated, batch_size=self.dataset_config.batch_size.config_option,
                          shuffle=self.dataset_config.shuffle.config_option, collate_fn=self.do_collate)

    def do_collate(self, batch_items: list):
        LoggerFacade.info(f"Doing collation with {', '.join(j for b in batch_items for j in b.keys())}.")
        assert len(batch_items) == 1
        batch_item = batch_items[0]

        assert isinstance(batch_item, dict), f"Batch items: {batch_item} was not recognized data type."
        LoggerFacade.info(f"Doing collation with {create_torch_size_log(batch_item[INPUT_IDS])} and "
                          f"{create_torch_size_log(batch_item[ATTENTION_MASK])}.")

        self._assert_batch_size_correct(batch_item)

        LoggerFacade.info(f"Finished doing collation: {create_torch_size_log(batch_item[INPUT_IDS])} "
                          f"and {create_torch_size_log(batch_item[ATTENTION_MASK])}.")
        return batch_item

    def _assert_batch_size_correct(self, batch_item):
        assert batch_item[INPUT_IDS].shape[0] == self.dataset_config.batch_size.config_option, \
            f"Batch item: {create_torch_size_log(batch_item[INPUT_IDS])} did not match batch size "\
            f"{self.dataset_config.batch_size.config_option}."

    @property
    def decorated(self) -> DecoratedDataset:
        return self._decorated
