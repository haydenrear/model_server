from typing import Optional

import torch.nn.init
from torch import nn

from drools_py.configs.config_models import VocabSize
from drools_py.loss.loss_types import LossTypes, LossTypesConfigOption
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory


class LossConfig(Config):

    def __init__(self,
                 initialization_type: LossTypes):
        self.initialization_type = initialization_type

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return LossConfig.update_override(
            LossConfig(LossTypes.CrossEntropy).to_self_dictionary(), kwargs)


import torch


class PenaltyTermCalculator:
    def __init__(self, vocab_size, padding_token, rolling_average_factor=0.9):
        self.padding_token = [padding_token] if isinstance(padding_token, int) else padding_token
        self.vocab_size = vocab_size
        self.rolling_average_factor = rolling_average_factor
        self.total_count = 1
        self.avg_word_probs = torch.zeros(vocab_size)

    def update_history(self, word_counts):
        self.total_count += 1

        # Update the average word probabilities
        self._set_padding_tokens(word_counts)
        self.avg_word_probs = (self.rolling_average_factor * self.avg_word_probs * self.total_count
                               + (1 - self.rolling_average_factor) * word_counts) / self.total_count

    def _set_padding_tokens(self, word_counts):
        for p in self.padding_token:
            word_counts[..., p] = 0.0
        return word_counts

    def compute_penalty_term(self):
        abs_value = torch.abs(self.total_count * self.avg_word_probs - self.avg_word_probs)
        abs_value = self._set_padding_tokens(abs_value)
        penalty_term = torch.sum(abs_value)
        return penalty_term

    def compute_difference_term(self, input_tensor, tgt=None, mask: Optional[torch.Tensor] = None):
        # Compute the difference term between the input tensor and the current average word probabilities
        assert input_tensor.size(-1) == self.vocab_size
        input_tensor = self._set_padding_tokens(input_tensor)
        self.avg_word_probs = self._set_padding_tokens(self.avg_word_probs)
        diff = torch.abs(input_tensor - self.avg_word_probs.expand(input_tensor.shape))
        avg_proba = torch.mean(diff, dim=-1)
        tgt = tgt.type(torch.int)
        for p in self.padding_token:
            try:
                avg_proba[tgt == int(p)] = 0.0
            except:
                try:
                    for p_ in p:
                        avg_proba[tgt == int(p_)] = 0.0
                except:
                    pass

        return torch.sum(avg_proba)



class LossConfigFactory(ConfigFactory):
    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        self.loss_config = loss_config

    def create(self, **kwargs):
        init_type = self.loss_config.initialization_type
        if init_type == LossTypes.CrossEntropy:
            return torch.nn.CrossEntropyLoss(**kwargs)
        elif init_type == LossTypes.KlDivergence:
            return torch.nn.KLDivLoss(**kwargs)
        else:
            raise ValueError(f"Unknown loss type: {init_type}")

    @classmethod
    def loss_config_factory(cls, loss_types_config_option: LossTypesConfigOption):
        return LossConfigFactory(
            LossConfig(loss_types_config_option.config_option)
        )


class Loss:

    def __init__(self, loss_factory: LossConfigFactory, **kwargs):
        self.loss_args = kwargs
        self.loss = loss_factory.create(**kwargs)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.loss(*args, **kwargs)
