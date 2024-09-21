import os

from datasets import load_dataset, load_from_disk

from drools_py.data.download_data.base_download_dataset import AbstractBaseDatasetDownloader
from drools_py.data.download_data.download_dataset_config import DownloadTorchDatasetConfig
from python_util.logger.logger import LoggerFacade


class HuggingfaceDatasetDownloader(AbstractBaseDatasetDownloader):

    def __init__(self, dataset_args: DownloadTorchDatasetConfig):
        self.dataset_args = dataset_args
        self.config = dataset_args
        self.dataset_dir = os.path.join(self.config.save_data_dir.config_option,
                                        self.config.dataset_uri.config_option)
        self.dataset = self.download_dataset(**dataset_args.args)
        assert self.dataset, "Dataset was not downloaded or loaded successfully."

    def download_dataset(self, **kwargs):
        try:
            if os.path.exists(self.dataset_dir):
                try:
                    loaded = load_from_disk(self.dataset_dir)
                    LoggerFacade.info(f"Loaded dataset from disk: {self.dataset_dir}")
                    return loaded
                except Exception as e:
                    LoggerFacade.error(f'Failed to download dataset {self.config.dataset_uri.config_option} with '
                                       f'error {e}.')
            dataset = load_dataset(self.config.dataset_uri.config_option, cache_dir=self.dataset_dir,
                                   streaming=self.config.streaming_mode.config_option,
                                   token="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",
                                   **(self.config.dataset_load_args.config_option
                                      if self.config.dataset_load_args.config_option is not None
                                      else {}))
            if not self.config.streaming_mode:
                try:
                    dataset.save_to_disk(self.dataset_dir)
                except Exception as e:
                    LoggerFacade.error(f'Failed to save {self.config.dataset_uri.config_option} to direcory: '
                                       f'{self.dataset_dir} with error {e}.')
            return dataset
        except Exception as e:
            LoggerFacade.error(f'Error loading huggingface dataset: {e}')
        return None

    def continue_download(self):
        self.dataset = self.download_dataset()
        return self.dataset


