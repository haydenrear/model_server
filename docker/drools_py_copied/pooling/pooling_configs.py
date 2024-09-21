import abc
from typing import Optional

import torch

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.pooling.pooling_types import PoolTypeConfigOption, PoolType


class PoolerConfig(Config):

    def __init__(self, pool_type: PoolTypeConfigOption):
        self.pool_type = pool_type



class PoolerFactory(ConfigFactory):
    def __init__(self, pooler_config: PoolerConfig):
        super().__init__(pooler_config)
        self.pooler_config = pooler_config

    def create(self, pool_dim: Optional[int] = None, **kwargs):
        if self.pooler_config.pool_type.config_option == PoolType.Mean:
            return lambda input_value, dim = None: torch.mean(input_value, dim=dim if dim is not None else pool_dim)
        elif self.pooler_config.pool_type.config_option == PoolType.NoPool:
            return lambda input_value, dim: input_value
