import abc
import time
from typing import Optional

import torch

from drools_py.model_ctx.model_context import ModelContextProcessor, ModelContextSubscriber
from python_di.configs.autowire import injectable
from python_util.logger.logger import LoggerFacade


class SoftmaxOutputModel(abc.ABC):
    @abc.abstractmethod
    def softmax_forward(self, input_ids: torch.Tensor,
                        attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass


class SoftmaxModelAdapter(ModelContextSubscriber):


    def update_arg(self, key: str, value):
        self.model = value
        assert isinstance(self.model, SoftmaxOutputModel)

    def set_values(self, model_context_processor: ModelContextProcessor):
        self.model_context_processor = model_context_processor
        self.model_context_processor.subscribe_to_key(self, 'softmax_model')

    def __call__(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        self.await_model_initialization()
        return self.model.softmax_forward(input_ids, attn_mask)

    def await_model_initialization(self):
        count = 0
        while self.model_context_processor is None or self.model is None:
            time.sleep(3)
            count += 1
            if count % 10 == 0:
                LoggerFacade.warn(f"Waiting for softmax model in softmax beam search adapter after {count * 3} "
                                  f"seconds.")


class SoftmaxBeamSearchAdapter(SoftmaxModelAdapter):

    def __init__(self):
        super().__init__()
        self.model: Optional[SoftmaxOutputModel] = None
        self.model_context_processor = None

    @injectable()
    def set_values(self, model_context_processor: ModelContextProcessor):
        super().set_values(model_context_processor)
        assert self.model_context_processor is not None


class MetropolisHastingsModel(SoftmaxModelAdapter):

    def __init__(self):
        super().__init__()
        self.model: Optional[SoftmaxOutputModel] = None
        self.model_context_processor = None

    @injectable()
    def set_values(self, model_context_processor: ModelContextProcessor):
        super().set_values(model_context_processor)
        assert self.model_context_processor is not None
