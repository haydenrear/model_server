import abc

from datasets import DatasetDict
from torch.utils.data import Dataset

from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import DatasetUri, SaveDatasetUri, DatasetLoadKwargs, StreamDataset
from drools_py.data.dataset.datasets.abstract_datasets import DatasetType, DataType, DatasetSourceType
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn


class DatasetProducer(ConfigFactory, abc.ABC):
    pass


class DownloadTorchDatasetConfig(DatasetProducer, Config):

    def __init__(self,
                 dataset_uri: DatasetUri,
                 save_data_dir: SaveDatasetUri,
                 data_set_type: DatasetType,
                 data_type: DataType,
                 dataset_load_args: DatasetLoadKwargs,
                 streaming_mode: StreamDataset,
                 dataset_ty: DatasetSourceType,
                 **kwargs):
        self.dataset_ty = dataset_ty
        self.dataset_load_args = dataset_load_args
        self.args = kwargs
        self.streaming_mode = streaming_mode
        self.data_type = data_type
        self.data_set_type = data_set_type
        self.save_data_dir = save_data_dir
        self.dataset_uri = dataset_uri
        DatasetProducer.__init__(self, self)

    @classmethod
    @autowire_fn()
    def download_dataset_config(cls,
                                config_type: ConfigType,
                                dataset_uri: DatasetUri,
                                save_data_dir: SaveDatasetUri,
                                data_set_type: DatasetType,
                                data_type: DataType,
                                dataset_load_args: DatasetLoadKwargs,
                                streaming_mode: StreamDataset,
                                dataset_ty: DatasetSourceType):
        return DownloadTorchDatasetConfig(dataset_uri, save_data_dir, data_set_type, data_type,
                                          dataset_load_args, streaming_mode, dataset_ty)

    def create(self, dataset_config, **kwargs):
        from drools_py.data.dataset.dataset_config import DatasetConfig
        dataset_config: DatasetConfig = dataset_config
        assert self.dataset_ty == DatasetSourceType.Huggingface, \
            "Only hf is implemented."
        from drools_py.data.download_data.huggingface_dataset_loader import HuggingfaceDatasetDownloader
        dataset_loader = HuggingfaceDatasetDownloader(self)
        downloaded_hf_dataset = dataset_loader.download_dataset()
        assert isinstance(downloaded_hf_dataset, Dataset | DatasetDict), \
            f"Dataset downloaded from {dataset_config} was not a torch dataset."
        return downloaded_hf_dataset

