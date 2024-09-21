import abc
from abc import ABC

from drools_py.configs.config import Config


class ConfigFactory(Config, abc.ABC):

    def __init__(self, config_of_item_to_create: Config = None):
        self.config_of_item_to_create = config_of_item_to_create

    @abc.abstractmethod
    def create(self, **kwargs):
        pass


class ModelConfigFactory(ConfigFactory, ABC):
    """
    Marker interface for config factory for entire model.
    """
    pass
