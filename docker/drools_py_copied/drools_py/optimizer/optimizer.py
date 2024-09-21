import os.path
from typing import Iterator, Optional

import torch.optim

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import LearningRate, WeightDecay, CheckpointPath
from drools_py.optimizer.optimizer_types import OptimizerType, OptimizerTypeConfigOption
from python_util.logger.logger import LoggerFacade


class OptimizerConfig(Config):

    def __init__(self,
                 learning_rate: LearningRate,
                 weight_decay: WeightDecay,
                 optimizer_type: OptimizerType,
                 checkpoint_path: Optional[CheckpointPath] = None):
        self.checkpoint_path = checkpoint_path
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type


class OptimizerConfigFactory(ConfigFactory):
    def __init__(self, optimizer_config: OptimizerConfig):
        super().__init__(optimizer_config)
        self.optimizer_config = optimizer_config

    def create(self, parameters: Iterator[torch.nn.Parameter], **kwargs):
        optimizer = None
        if self.optimizer_config.optimizer_type == OptimizerType.AdamW:
            optimizer = torch.optim.AdamW(parameters, lr=self.optimizer_config.learning_rate.config_option,
                                          weight_decay=self.optimizer_config.weight_decay.config_option)
        elif self.optimizer_config.optimizer_type == OptimizerType.Adam:
            optimizer = torch.optim.Adam(parameters, lr=self.optimizer_config.learning_rate.config_option,
                                         weight_decay=self.optimizer_config.weight_decay.config_option)
        elif self.optimizer_config.optimizer_type == OptimizerType.RAdam:
            optimizer = torch.optim.RAdam(parameters, lr=self.optimizer_config.learning_rate.config_option)

        if optimizer is None:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_config.optimizer_type}")
        if self.optimizer_config.checkpoint_path is not None:
            LoggerFacade.debug(f"Optimizer checkpoint path is {self.optimizer_config.checkpoint_path} Attempting "
                               f"to load.")
            if os.path.exists(self.optimizer_config.checkpoint_path.config_option):
                try:
                    loaded = torch.load(self.optimizer_config.checkpoint_path.config_option)
                    optimizer = loaded['optimizer_state_dict']
                    optimizer.load_state_dict(optimizer)
                    return optimizer
                except Exception as e:
                    LoggerFacade.error("Error loading checkpoint!")
                    raise e
            else:
                LoggerFacade.error(f"Optimizer checkpoint {self.optimizer_config.checkpoint_path} was provided but "
                                   f"did not exist. No checkpoint was loaded for optimizer.")

        return optimizer

    @classmethod
    def optimizer_config_factory(cls, optimizer_config_type: OptimizerTypeConfigOption,
                                 learning_rate: LearningRate, weight_decay: WeightDecay,
                                 checkpoint_path: Optional[CheckpointPath] = None):
        return OptimizerConfigFactory(
            OptimizerConfig(
                optimizer_type=optimizer_config_type.config_option,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                checkpoint_path=checkpoint_path
            )
        )