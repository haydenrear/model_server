import abc
from typing import Optional

import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, \
    CosineAnnealingLR, LRScheduler, ConstantLR

from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory
from drools_py.configs.config_models import StepSize, LearningRateGamma, NumEpochs, MaxLearningRate, LearningRateMode, \
    WarmupIterationsLr, NumIterationsFirstRestartLr, StepsPerEpochLr, EpochMultiplierRestart, MinLearningRate
from drools_py.learning_rate.cosine_learning_rate_warmup import CosineWarmupScheduler
from drools_py.learning_rate.learning_rate_types import LearningRateSchedulerType, LearningRateSchedulerTypeConfigOption
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope
import injector


class LearningRateStepper(abc.ABC):
    def do_learning_rate(self, epoch: Optional = None):
        pass


class LearningRateStepperConfig(Config):

    def __init__(self,
                 step_size: StepSize,
                 gamma: LearningRateGamma,
                 num_epochs: NumEpochs,
                 max_lr: MaxLearningRate,
                 mode: LearningRateMode,
                 num_iterations_first_restart: NumIterationsFirstRestartLr,
                 num_iterations_warmup: WarmupIterationsLr,
                 steps_per_epoch: StepsPerEpochLr,
                 epoch_multiplier_restart: EpochMultiplierRestart,
                 min_lr: MinLearningRate):
        self.min_lr = min_lr
        self.epoch_multiplier_restart = epoch_multiplier_restart
        self.steps_per_epoch = steps_per_epoch
        self.num_iterations_warmup = num_iterations_warmup
        self.num_iterations_first_restart = num_iterations_first_restart
        self.mode = mode
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.step_size = step_size
        self.gamma = gamma


class NoOp(ConstantLR):
    def __init__(self, optimizer: torch.optim.Optimizer):
        super().__init__(optimizer=optimizer, factor=1)


class LearningRateStepperConfigFactory(ConfigFactory):

    def __init__(self, config: LearningRateStepperConfig,
                 learning_rate_stepper_type: LearningRateSchedulerTypeConfigOption):
        super().__init__(config)
        self.learning_rate_stepper_type = learning_rate_stepper_type
        self.config = config

    def create(self, optimizer, **kwargs):
        if self.learning_rate_stepper_type == LearningRateSchedulerType.NoOp:
            return NoOp(optimizer)
        if self.learning_rate_stepper_type == LearningRateSchedulerType.OneCycleLR:
            return OneCycleLR(optimizer, self.config.max_lr.config_option,
                              self.config.num_epochs * self.config.steps_per_epoch,
                              self.config.num_epochs.config_option, self.config.steps_per_epoch.config_option)
        elif self.learning_rate_stepper_type == LearningRateSchedulerType.StepLR:
            return StepLR(optimizer, self.config.step_size.config_option, self.config.gamma.config_option)
        elif self.learning_rate_stepper_type == LearningRateSchedulerType.ReduceLROnPlateau:
            return ReduceLROnPlateau(optimizer, self.config.mode.config_option.name)
        elif self.learning_rate_stepper_type == LearningRateSchedulerType.CosineAnnealingLR:
            return CosineAnnealingLR(optimizer, self.config.num_epochs * self.config.steps_per_epoch,
                                     self.config.min_lr.config_option)
        elif self.learning_rate_stepper_type == LearningRateSchedulerType.CosineAnnealingWarmRestarts:
            return {
                'scheduler': CosineAnnealingWarmRestarts(optimizer, self.config.steps_per_epoch.config_option,
                                                         self.config.epoch_multiplier_restart.config_option),
                'name': 'train/lr',
                'interval': 'step',
                'frequency': 1
            }
        elif self.learning_rate_stepper_type == LearningRateSchedulerType.CosineWarmup:
            return CosineWarmupScheduler(optimizer, self.config.num_iterations_warmup.config_option,
                                         self.config.num_epochs * self.config.steps_per_epoch)
        # elif self.config.learning_rate_stepper_type == LearningRateSchedulerType.SWALR:
        #     return SWALR(optimizer, )

    @classmethod
    def lr_config_factory(cls, lr_ty: LearningRateSchedulerTypeConfigOption, config: LearningRateStepperConfig):
        return LearningRateStepperConfigFactory(
            config, lr_ty
        )
