import abc
from typing import Optional

from codegen.generated_config_models import BatchSizeRecursiveCat, RecursiveCatEmbeddingSize, SequenceLengthRecursiveCat
from drools_py.configs.config_models import BatchSize, EmbeddingSize, SequenceLength

from drools_py.configs.config import Config, ConfigType
import torch

from python_di.inject.profile_composite_injector.inject_context_di import autowire_fn
from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.padding import pad_add_end_to_match
from python_util.torch_utils.pytorch_util import create_torch_size_log


class BaseSplitEmitConfig(Config):
    def __init__(self, batch_size: BatchSize,
                 embedding_size: EmbeddingSize,
                 seq_len: SequenceLength):
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.batch_size = batch_size


class BaseSplitEmit:

    def __init__(self, split: BaseSplitEmitConfig):
        self.split = split
        self.state = self._reset_state()
        self.encoder_state = self._reset_state()

        self.split_size = self._get_split_size(split)

        self.src_mask = self._reset_attn()
        self.tgt_mask = self._reset_attn()

        self.encoder_state_pointer = 0
        self.state_pointer = 0
        self.src_mask_pointer = 0
        self.tgt_mask_pointer = 0

    @abc.abstractmethod
    def _get_split_size(self, split) -> int:
        pass

    @property
    @abc.abstractmethod
    def split_state_idx(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def split_attn_idx(self) -> int:
        pass

    def split_chunks(self, curr_batch_size, prev_decoder_states, encoder_states, src_mask, tgt_mask):

        attn_mask_chunks = None
        key_attn_mask_chunks = None
        encoder_chunks = None
        prev_decoder_state_chunks = None

        encoder_states, prev_decoder_states, src_mask, tgt_mask = self._pad_to_split(
            encoder_states, prev_decoder_states, src_mask, tgt_mask)

        num_split = curr_batch_size // self.split_size
        if num_split == 1 and curr_batch_size > self.split_size:
            num_split = 2

        if prev_decoder_states is not None:
            prev_decoder_state_chunks = torch.tensor_split(prev_decoder_states, max(1, num_split), self.split_state_idx)
        if encoder_states is not None:
            encoder_chunks = torch.tensor_split(encoder_states, max(1, num_split), self.split_state_idx)

        if tgt_mask is not None:
            attn_mask_chunks = torch.chunk(tgt_mask, max(1, num_split), self.split_attn_idx)
        if src_mask is not None:
            key_attn_mask_chunks = torch.tensor_split(src_mask, max(1, num_split), self.split_attn_idx)

        return prev_decoder_state_chunks, encoder_chunks, key_attn_mask_chunks, attn_mask_chunks

    def do_register_emit(self, prev_decoder_states, encoder_states, src_mask, tgt_mask, truncate_tensor: bool = False) \
            -> list[(torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor])]:
        """
        Inputs the states and then returns a list of batches of size self.batch_size. There is an internal state such
        that if when registering you do not hit the minimum size, then it will return an empty list.
        :param prev_decoder_states:
        :param encoder_states:
        :param src_mask:
        :param tgt_mask:
        :param truncate_tensor:
        :return:
        """

        self._do_assert_emit_input(encoder_states, prev_decoder_states, src_mask, tgt_mask)

        curr_batch_size = prev_decoder_states.shape[self.split_state_idx]

        prev_decoder_chunks, encoder_state_chunks, src_chunks, tgt_chunks = self.split_chunks(
            curr_batch_size, prev_decoder_states, encoder_states, src_mask, tgt_mask)

        out_states = []

        for i, encoder_chunk in enumerate(prev_decoder_chunks if prev_decoder_chunks is not None else encoder_states):
            out_encoder_state, out_state = self._get_set_states(encoder_chunk, encoder_state_chunks, i,
                                                                prev_decoder_chunks)

            out_mask_state = self._get_tgt_chunks(encoder_chunk, i, tgt_chunks)

            out_src_mask_state = self._get_src_chunks(encoder_chunk, i, src_chunks)

            if out_state is not None or out_encoder_state is not None:
                out_states.append((out_state, out_encoder_state, out_mask_state, out_src_mask_state))

        return out_states

    def _get_set_states(self, encoder_chunk, encoder_state_chunks, i, prev_decoder_chunks):
        if prev_decoder_chunks is not None:
            out_state, self.state_pointer, self.state = self._add_to_state(
                self.state, encoder_chunk, self.state_pointer)
        else:
            out_state = None
        if encoder_state_chunks is not None:
            out_encoder_state, self.encoder_state_pointer, self.encoder_state = self._add_to_state(
                self.encoder_state, encoder_state_chunks[i], self.encoder_state_pointer)
        else:
            out_encoder_state = None
        return out_encoder_state, out_state

    def _get_tgt_chunks(self, encoder_chunk, i, tgt_chunks):
        if tgt_chunks is not None:
            out_mask_state, self.tgt_mask_pointer, self.tgt_mask \
                = self._add_to_attn_mask(tgt_chunks[i], self.tgt_mask, self.tgt_mask_pointer)
        else:
            out_mask_state, self.tgt_mask_pointer, self.tgt_mask \
                = self._add_to_attn_mask(torch.ones([encoder_chunk.shape[1], encoder_chunk.shape[0]]),
                                         self.tgt_mask, self.tgt_mask_pointer)
        return out_mask_state

    def _get_src_chunks(self, encoder_chunk, i, src_chunks):
        if src_chunks is not None:
            out_src_mask_state, self.src_mask_pointer, self.src_mask \
                = self._add_to_attn_mask(src_chunks[i], self.src_mask, self.src_mask_pointer)
        else:
            out_src_mask_state, self.src_mask_pointer, self.src_mask \
                = self._add_to_attn_mask(torch.ones([encoder_chunk.shape[1], encoder_chunk.shape[0]]),
                                         self.src_mask, self.src_mask_pointer)
        return out_src_mask_state

    def _do_assert_emit_input(self, encoder_states, prev_decoder_states, src_mask, tgt_mask):
        LoggerFacade.info(f"Encoder states: {create_torch_size_log(encoder_states)}, "
                          f"Prev encoder states: {create_torch_size_log(prev_decoder_states)}, "
                          f"Src mask: {create_torch_size_log(src_mask)}, "
                          f"Tgt mask: {create_torch_size_log(tgt_mask)}.")
        self._assert_state_seq_len(encoder_states, prev_decoder_states)
        self._assert_state_batch_size(encoder_states, prev_decoder_states)
        self._assert_mask_state_batch(prev_decoder_states, src_mask)
        self._assert_mask_state_seq_len(prev_decoder_states, src_mask)
        self._assert_mask_state_seq_len(prev_decoder_states, tgt_mask)
        self._assert_mask_state_batch(prev_decoder_states, tgt_mask)
        self._assert_mask_state_batch(encoder_states, src_mask)
        self._assert_mask_state_seq_len(encoder_states, tgt_mask)
        self._assert_mask_state_seq_len(encoder_states, src_mask)
        self._assert_mask_state_batch(encoder_states, tgt_mask)

    def _assert_state_batch_size(self, encoder_states, prev_decoder_states):
        assert (prev_decoder_states is None or encoder_states is None
                or prev_decoder_states.shape[1] == encoder_states.shape[1])

    def _assert_state_seq_len(self, encoder_states, prev_decoder_states):
        assert (prev_decoder_states is None or encoder_states is None
                or prev_decoder_states.shape[0] == encoder_states.shape[0])

    def _assert_mask_state_seq_len(self, encoder_states, src_mask):
        assert (encoder_states is None or src_mask
                is None or src_mask.shape[1] == encoder_states.shape[0])

    def _assert_mask_state_batch(self, encoder_states, tgt_mask):
        assert (encoder_states is None or tgt_mask
                is None or tgt_mask.shape[0] == encoder_states.shape[1])

    def _pad_to_split(self, encoder_states, prev_decoder_states, src_mask, tgt_mask):
        prev_decoder_states = self.pad_state(prev_decoder_states)
        encoder_states = self.pad_state(encoder_states)
        src_mask = self.pad_mask(src_mask)
        tgt_mask = self.pad_mask(tgt_mask)
        return encoder_states, prev_decoder_states, src_mask, tgt_mask

    @abc.abstractmethod
    def pad_mask(self, src_mask):
        pass

    @abc.abstractmethod
    def pad_state(self, prev_decoder_states):
        pass

    def _assert_seq_len_size(self, prev_decoder_states, size):
        assert prev_decoder_states is None or size <= self.split_size, \
            (f"Prev decoder state of size {size} was greater than max seq len: "
             f"{self.split_size}. Prev decoder states producing issue are {prev_decoder_states.shape} and "
             f"split size is {self.split_size}, and in {type(self).__name__}.")

    def _mod_batch_ptr(self, new_batch_pointer):
        if new_batch_pointer >= self.split_size:
            return new_batch_pointer % self.split_size
        elif new_batch_pointer == self.split_size:
            return 0
        else:
            return new_batch_pointer

    def _reset_attn(self):
        return torch.zeros([self.split.seq_len.config_option,
                            self.split.batch_size.config_option]).T

    def _reset_state(self):
        return torch.zeros([self.split.seq_len.config_option,
                            self.split.batch_size.config_option,
                            self.split.embedding_size.config_option])

    def permute_to_add_to(self, prev_decoder_states):
        return prev_decoder_states if prev_decoder_states is not None else None

    def permute_post(self, prev_decoder_states):
        return prev_decoder_states if prev_decoder_states is not None else None

    def permute_attn(self, attn):
        return attn if attn is not None else None

    def permute_post_attn(self, attn):
        return attn if attn is not None else None
