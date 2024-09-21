from drools_py.configs.config import ConfigType
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import BatchSize, ShuffleData
from drools_py.data.dataset.dataset_config import DatasetConfig, DatasetConfigFactory
from drools_py.data.dataset.dataset_decorator.decorated_dataset import DecoratedDatasetConfigFactory
from drools_py.data.dataset.datasets.abstract_datasets import DataIterDecoratedFactoryConfig
from drools_py.data.dataset.datasets.torch_dataset import TorchDatasetDecorator
from drools_py.data.download_data.download_dataset_config import DownloadTorchDatasetConfig
from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn


class HuggingfaceDatasetConfig(DatasetConfig):

    def __init__(self,
                 config_type: ConfigType,
                 load_config: DownloadTorchDatasetConfig,
                 batch_size: BatchSize,
                 shuffle: ShuffleData,
                 decorated: DecoratedDatasetConfigFactory):
        super().__init__(config_type, load_config, batch_size, shuffle, decorated)
        self.config_type = config_type

    @classmethod
    @autowire_fn()
    def dataset_config(cls,
                       config_type: ConfigType,
                       load_config: DownloadTorchDatasetConfig,
                       batch_size: BatchSize,
                       shuffle: ShuffleData,
                       decorated: DecoratedDatasetConfigFactory):
        return HuggingfaceDatasetConfig(config_type, load_config, batch_size, shuffle,
                                        decorated)


class HuggingfaceDatasetFactory(DatasetConfigFactory):

    def __init__(self, config_of_item_to_create: HuggingfaceDatasetConfig):
        super().__init__(config_of_item_to_create)
        self.config: HuggingfaceDatasetConfig = config_of_item_to_create

    def create(self, **kwargs) -> TorchDatasetDecorator:
        dataset_config = self.config
        return TorchDatasetDecorator(dataset_config)


    @classmethod
    @autowire_fn()
    def dataset_factory(cls, config_type: ConfigType, config: DatasetConfig):
        assert isinstance(config, HuggingfaceDatasetConfig)
        return HuggingfaceDatasetFactory(config)

