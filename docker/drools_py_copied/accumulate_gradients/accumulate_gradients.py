from drools_py.accumulate_gradients.accumulate_gradients_types import AccumulateGradientsTypes, \
    AccumulateGradientsConfigOption
from drools_py.configs.config import Config
from drools_py.configs.config_models import BatchSize, AccumulateGradientsBatchSize
from drools_py.lightning_trainer_args_decorator.trainer_args_decorator import LightningTrainerArgsDecorator


class AccumulateGradientsConfig(Config):

    def __init__(self,
                 accumulate_grads: AccumulateGradientsTypes,
                 batch_size: AccumulateGradientsBatchSize):
        self.batch_size = batch_size
        self.accumulate_grads = accumulate_grads


class AccumulateGradientsConfigFactory(LightningTrainerArgsDecorator):
    def __init__(self, accumulate_gradients_config: AccumulateGradientsConfig):
        super().__init__(accumulate_gradients_config)
        self.accumulate_gradients_config = accumulate_gradients_config

    def create(self, trainer_args: dict, **kwargs):
        if self.accumulate_gradients_config.accumulate_grads == AccumulateGradientsTypes.AccumulateGradients:
            accumulate_gradient_batch_size = self.accumulate_gradients_config.batch_size
            trainer_args['accumulate_grad_batches'] = accumulate_gradient_batch_size.config_option
            return trainer_args
        else:
            return trainer_args

    @classmethod
    def accumulate_gradients(cls, accumulate_gradients_batch: AccumulateGradientsBatchSize,
                             accumulate_gradients: AccumulateGradientsConfigOption):
        return AccumulateGradientsConfigFactory(
            AccumulateGradientsConfig(accumulate_gradients.config_option, accumulate_gradients_batch)
        )
