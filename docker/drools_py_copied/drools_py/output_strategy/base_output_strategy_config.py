import injector

from drools_py.configs.config import Config
from drools_py.configs.config_models import TopK, TopP, Temperature, BeamWidth, NumSamples, VocabSize, StartToken, \
    EosToken
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope


@component(profile=['test', 'validation'], scope=profile_scope)
class OutputStrategyConfig(Config):
    @injector.inject
    def __init__(self, top_k: TopK,
                 top_p: TopP,
                 temperature: Temperature,
                 beam_width: BeamWidth,
                 num_samples: NumSamples,
                 vocab_size: VocabSize,
                 start_token: StartToken,
                 eos_token_idx: EosToken):
        self.eos_token_idx = eos_token_idx
        self.start_token = start_token
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.beam_width = beam_width
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k


