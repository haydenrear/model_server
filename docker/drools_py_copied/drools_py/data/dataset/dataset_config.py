import abc

from drools_py.configs.config import Config, ConfigType
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import BatchSize, ShuffleData
from drools_py.data.dataset.dataset_decorator.decorated_dataset import DecoratedDatasetConfigFactory
from drools_py.data.dataset.datasets.abstract_datasets import DataIterDecoratedFactoryConfig
from drools_py.data.download_data.download_dataset_config import DownloadTorchDatasetConfig



class DatasetConfig(Config):

    def __init__(self,
                 config_type: ConfigType,
                 load_config: DownloadTorchDatasetConfig,
                 batch_size: BatchSize,
                 shuffle: ShuffleData,
                 decorated: DecoratedDatasetConfigFactory):
        self.decorated = decorated
        self.config_type = config_type
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.load_config = load_config


class DatasetConfigFactory(ConfigFactory):

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config)
        self.config = dataset_config

    @abc.abstractmethod
    def create(self, **kwargs) -> DatasetConfig:
        pass
