import abc
from typing import Union, Iterable, Dict, Any, Optional, Callable

import torch.optim

from drools_py.configs.config import ConfigFactory
from drools_py.optimizer.optimizer import OptimizerConfig
from drools_py.optimizer.optimizer_types import OptimizerType


class OptimizerDecoratorConfig(OptimizerConfig):

    def __init__(self,
                 params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
                 learning_rate: float = 3e-5,
                 weight_decay: float = 1e-5,
                 defaults: Dict[str, Any] = None):
        super().__init__(learning_rate, weight_decay, OptimizerType.Wrapped)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.params = params
        self.defaults = defaults

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return OptimizerDecoratorConfig.update_override(OptimizerDecoratorConfig(
            [torch.zeros(0, requires_grad=True)],
            learning_rate=3e-5,
            weight_decay=1e-5
        ), kwargs)


class OptimizerDecorator(torch.optim.Optimizer, abc.ABC):

    def __init__(self, config: OptimizerDecoratorConfig):
        super().__init__(config.params,
                         config.defaults if config.defaults else {})
        self.delegate: Optional[torch.optim.Optimizer] = None
        self.config = config

    @abc.abstractmethod
    def wrap_optimizer(self, parameters, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        pass

    def do_or_raise(self, to_do):
        if not self.delegate:
            raise ValueError("Optimizer not set.")
        else:
            return to_do()

    def __setstate__(self, state):
        return self.do_or_raise(lambda: self.delegate.__setstate__(state))

    def register_step_pre_hook(self, hook: Callable[..., None]):
        return self.do_or_raise(lambda: self.delegate.register_step_pre_hook(hook))

    def register_step_post_hook(self, hook: Callable[..., None]):
        return self.do_or_raise(lambda: self.delegate.register_step_post_hook(hook))

    def state_dict(self):
        return self.do_or_raise(lambda: self.delegate.state_dict())

    def load_state_dict(self, state_dict):
        return self.do_or_raise(lambda: self.delegate.load_state_dict(state_dict))

    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        step = self.do_or_raise(lambda: self.delegate.step(closure))
        return step

    def zero_grad(self, set_to_none: bool = ...) -> None:
        return self.do_or_raise(lambda: self.delegate.zero_grad(set_to_none))


class OptimizerDecoratorConfigFactory(ConfigFactory, OptimizerDecoratorConfig, abc.ABC):

    def __init__(self, config_of_item_to_create: OptimizerDecoratorConfig):
        ConfigFactory.__init__(self, config_of_item_to_create)
        OptimizerDecoratorConfig.__init__(self,
                                          config_of_item_to_create.params,
                                          config_of_item_to_create.learning_rate,
                                          config_of_item_to_create.weight_decay,
                                          config_of_item_to_create.defaults)

    @abc.abstractmethod
    def create(self, optimizer, parameters, **kwargs) -> OptimizerDecorator:
        pass
