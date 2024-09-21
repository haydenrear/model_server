from drools_py.configs.config import Config
from drools_py.configs.config_models import GradientClippingValue
from drools_py.gradient_clipping.gradient_clipping_types import GradientClippingTypes, GradientClippingConfigOption
from drools_py.lightning_trainer_args_decorator.trainer_args_decorator import LightningTrainerArgsDecorator


class GradientClippingConfig(Config):

    def __init__(self,
                 gradient_clip_algo: GradientClippingTypes,
                 gradient_clip_norm: GradientClippingValue):
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_algo = gradient_clip_algo


class GradientClippingConfigFactory(LightningTrainerArgsDecorator):
    def __init__(self, gradient_clipping_config: GradientClippingConfig):
        super().__init__(gradient_clipping_config)
        self.gradient_clipping_config = gradient_clipping_config

    def create(self, trainer_args: dict, **kwargs):
        if self.gradient_clipping_config.gradient_clip_algo != GradientClippingTypes.SkipGradientClipping:
            gradient_clip = self.gradient_clipping_config.gradient_clip_norm
            init_type = self.gradient_clipping_config.gradient_clip_algo
            trainer_args['gradient_clip_val'] = gradient_clip.config_option
            trainer_args['gradient_clip_algorithm'] = init_type.name
            return trainer_args
        else:
            return trainer_args

    @classmethod
    def gradient_clipping(cls, gradient_clipping_config_option: GradientClippingConfigOption,
                          gradient_clipping_value: GradientClippingValue):
        return GradientClippingConfigFactory(
            GradientClippingConfig(gradient_clipping_config_option.config_option,
                                   gradient_clipping_value)
        )