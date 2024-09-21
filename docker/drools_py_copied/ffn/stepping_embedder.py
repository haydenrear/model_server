import dataclasses

import torch
from torch import nn

from drools_py.configs.config import Config
from drools_py.configs.config_models import EmbeddingSize, Divisor, OutDim, NSteps
from python_di.configs.prototype import prototype_scope_bean, prototype_factory


def get_steps(n_steps: int, divisor: int, embedding_size: int):
    steps = []
    assert embedding_size % divisor == 0, \
        f"The embedding size {embedding_size} in recursive cat was not divisible by the divisor {divisor}."
    step = embedding_size - (embedding_size // divisor)
    step = step / n_steps
    for i in reversed(range(n_steps)):
        next_step = (embedding_size // divisor) + (i * step)
        next_step = int(min(next_step if next_step % 2 == 0 else next_step + 1, embedding_size))
        steps.append(next_step)
    return steps


@prototype_scope_bean()
class SteppingEmbedderConfig(Config):

    @prototype_factory()
    def __init__(self, embedding_size: EmbeddingSize, divisors: Divisor,
                 n_embedding_steps: NSteps, output_dim: OutDim):
        self.output_dim = output_dim
        self.n_embedding_steps = n_embedding_steps
        self.embedding_size = embedding_size
        self.divisors = divisors


class SteppingEmbedder(torch.nn.Module):

    def __init__(self, stepping_embedder: SteppingEmbedderConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stepping_embedder = stepping_embedder
        self.sequence = self.create_embedding_sequence(stepping_embedder.embedding_size.config_option,
                                                       stepping_embedder.divisors.config_option,
                                                       stepping_embedder.n_embedding_steps.config_option,
                                                       stepping_embedder.output_dim.config_option)

    def forward(self, x) -> torch.Tensor:
        return self.sequence(x)

    @staticmethod
    def create_embedding_sequence(embedding_size: int,
                                  divisor: int,
                                  n_embedding_steps: int,
                                  output_dim: int) -> torch.nn.Sequential:
        """
        # TODO: split out into a stepping embedding layer - could be used for VAE also.
        The embedding can be multiple fully connected layers, which will have relu in between. In the case that
        there are multiple layers, it steps to the embedding with each layer.
        :return:
        """
        if n_embedding_steps == 1:
            return torch.nn.Sequential(*[nn.Linear(embedding_size, output_dim),
                                         torch.nn.ReLU()])
        embedding_sequence = [nn.Linear(embedding_size, embedding_size),
                              torch.nn.ReLU()]
        steps = get_steps(n_embedding_steps, divisor, embedding_size)
        prev_step = embedding_size
        divised_embed = embedding_size // divisor
        step = divised_embed
        for step in steps:
            embedding_sequence.append(nn.Linear(
                prev_step,
                step
            ))
            embedding_sequence.append(torch.nn.ReLU())
            prev_step = step
        if step != divised_embed:
            embedding_sequence.append(nn.Linear(
                step,
                divised_embed
            ))
            embedding_sequence.append(torch.nn.ReLU())

        return torch.nn.Sequential(*embedding_sequence)
