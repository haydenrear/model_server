import injector

import torch

from drools_py.configs.config_models import TopK, TopP, Temperature, BeamWidth, EosToken, StartToken, VocabSize, \
    NumSamples, StepSize
from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig
from drools_py.output_strategy.softmax_output_model import MetropolisHastingsModel
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope


def metropolis_hastings_torch(p, q_draw, stepsize, n_samples, x_init):
    x = x_init
    samples = [x]
    for _ in range(n_samples):
        x_star = q_draw(x, stepsize)
        alpha = min(1, p(x_star) * q_draw(x_star, x, stepsize) / (p(x) * q_draw(x, x_star, stepsize)))
        if torch.rand(1) < alpha:
            x = x_star
        samples.append(x)
    return torch.stack(samples)


@component(profile=['test', 'validation'], scope=profile_scope)
class MetropolisHastingsSamplingConfig(OutputStrategyConfig):


    @injector.inject
    def __init__(self, top_k: TopK, top_p: TopP, temperature: Temperature, beam_width: BeamWidth,
                 num_samples: NumSamples, vocab_size: VocabSize, start_token: StartToken, eos_token_idx: EosToken,
                 step_size: StepSize):
        """
        :param model: Softmaxed probabilities of index, given sequence of indices.
        :param vocabulary:
        :param step_size:
        :param top_k:
        :param top_p:
        :param temperature:
        :param beam_width:
        :param num_samples:
        :param vocab_size:
        :param start_token:
        :param eos_token_idx:
        """
        super().__init__(top_k, top_p, temperature, beam_width, num_samples, vocab_size, start_token, eos_token_idx)
        self.step_size = step_size


class MetropolisHastingsSampling(OutputStrategy):

    def __init__(self, output_strategy_config: MetropolisHastingsSamplingConfig):
        super().__init__(output_strategy_config)
        self.model = MetropolisHastingsModel()
        self.config = output_strategy_config

    def p(self, x_star, s):
        """
        Given a sequence s and a candidate next word x_star,
        return the probability of x_star given s using the LLM.
        """
        return self.model(s)[x_star]

    def q_draw(self, x):
        """
        Given a current word x, propose a new word based on the model's prediction.
        """
        probabilities = self.model(x)  # Assumes model returns logits over the vocabulary
        proposed_word = torch.multinomial(probabilities, 1)  # Sample from the distribution
        return proposed_word

    def sample(self, x_init, s):
        """
        :param x_init:
        :param s:
        :return:
        """
        x = x_init
        samples = [x]
        for _ in range(self.output_strategy_config.num_samples.config_option):
            x_star = self.q_draw(x)
            alpha = min(1, self.p(x_star, s) * self.q_draw(x_star) /
                        (self.p(x, s) * self.q_draw(x_star)))
            if torch.rand(1) < alpha:
                x = x_star
            samples.append(x)
        return torch.stack(samples)
