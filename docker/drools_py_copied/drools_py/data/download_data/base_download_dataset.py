import abc
import os

from drools_py.hash.sha256_hash import compute_sha256_hash, get_git_hash


class AbstractBaseDatasetDownloader(abc.ABC):

    @abc.abstractmethod
    def download_dataset(self):
        pass

    @abc.abstractmethod
    def continue_download(self):
        pass


class AbstractGitDatasetDownloader(abc.ABC):

    @staticmethod
    def is_file_downloaded(filename_dir: str, dataset_directory: str) -> bool:
        if os.path.isdir(filename_dir):
            return True
        hash_calculated = compute_sha256_hash(os.path.join(dataset_directory, filename_dir))
        return hash_calculated == get_git_hash(os.path.join(dataset_directory, filename_dir))
