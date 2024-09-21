import torch
from transformers import BeamSearchScorer, GenerationMixin, GenerationConfig
from transformers.utils.generic import ModelOutput

from drools_py.configs.config_models import SequenceLength, BatchSize, TopK, TopP, Temperature, BeamWidth, StartToken, \
    EosToken, NumSamples, VocabSize, Device
from drools_py.data.constant import INPUT_IDS
from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig

from drools_py.output_strategy.softmax_output_model import SoftmaxBeamSearchAdapter
from python_di.configs.component import component
from python_di.inject.profile_composite_injector.composite_injector import profile_scope
import injector


@component(profile=['test', 'validation'], scope=profile_scope)
class BeamSearchOutputStrategyConfig(OutputStrategyConfig):
    @injector.inject
    def __init__(self, sequence_len: SequenceLength, top_k: TopK, top_p: TopP, temperature: Temperature,
                 beam_width: BeamWidth, num_samples: NumSamples, vocab_size: VocabSize, start_token: StartToken,
                 eos_token_idx: EosToken, device: Device):
        super().__init__(top_k, top_p, temperature, beam_width, num_samples, vocab_size, start_token, eos_token_idx)
        self.device = device
        self.seq_len = sequence_len


class BeamSearchOutput(ModelOutput):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits


class BeamSearch(OutputStrategy, GenerationMixin):

    def __init__(self, config: BeamSearchOutputStrategyConfig):
        super().__init__(config)
        self.config = config
        self.softmax_model_producer: SoftmaxBeamSearchAdapter = SoftmaxBeamSearchAdapter()
        self.beam_width = config.beam_width.config_option
        self.beam_scorer = BeamSearchScorer(BatchSize().config_option,
                                            self.beam_width, config.device.config_option)
        self.max_len = config.seq_len.config_option
        self.eos_idx = config.eos_token_idx.config_option
        self.generation_config = GenerationConfig(pad_token_id=self.eos_idx,
                                                  eos_token_id=self.eos_idx)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        return {INPUT_IDS: input_ids}

    def sample(self, model_inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        inputs = model_inputs.T
        input_ids = (inputs.unsqueeze(0).expand(self.beam_width, inputs.shape[0], inputs.shape[1])
                     .reshape(self.beam_width * inputs.shape[0], inputs.shape[1]))
        return self.beam_search(
            input_ids,
            self.beam_scorer,
            output_scores=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
            output_attentions=False,
            max_length=min(self.max_len, model_inputs.shape[0] + 1)
        )

    def __call__(self, input_ids, *args, **kwargs):
        with torch.no_grad():
            return BeamSearchOutput(self.softmax_model_producer(input_ids, torch.ones_like(input_ids))
                                    .transpose(0, 1))
