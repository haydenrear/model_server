import abc




class ArchitectureContext(abc.ABC):

    @abc.abstractmethod
    def run_config_change_engine(self):
        pass

    @abc.abstractmethod
    def register_config_change(self, config_id, key, value):
        pass

    @abc.abstractmethod
    async def await_config_change_engine(self):
        pass
