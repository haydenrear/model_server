import abc
import typing
from typing import Optional

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.context import ArchitectureContext
from drools_py.configs.evolvable_module import EvolvableModule


class DecoratableModification(abc.ABC):

    @abc.abstractmethod
    def decorate_config(self,
                        to_decorate_config: typing.Union[Config, ConfigFactory]) -> typing.Union[Config, ConfigFactory]:
        pass

    @abc.abstractmethod
    def decorate_module(self, emit_forward: EvolvableModule) -> EvolvableModule:
        pass


class ArchitectureModification(abc.ABC):

    @abc.abstractmethod
    async def deploy_modification(self, parent_config: Config,
                                  architecture_context: ArchitectureContext,
                                  children_config: Optional[list[Config]] = None) \
            -> DecoratableModification:
        """
        :param parent_config:
        :param architecture_context:
        :param children_config: When the modification depends
        :return:
        """
        pass

    @abc.abstractmethod
    def register_change(self, architecture_context: ArchitectureContext):
        pass

