import abc

from drools_py.configs.config_factory import ConfigFactory


class LightningTrainerArgsDecorator(ConfigFactory, abc.ABC):

    @abc.abstractmethod
    def create(self, in_args: dict, **kwargs) -> dict:
        pass