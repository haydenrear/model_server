from typing import Optional

import torch

from drools_py.batch_split.base_split_emit import BaseSplitEmit
from python_util.torch_utils.padding import pad_add_end_to_match


class SplitSeqEmit(BaseSplitEmit):

    def _assert_seq_len_size(self, prev_decoder_states, size):
        assert prev_decoder_states is None or size <= self.split.batch_size.config_option, \
            (f"Prev decoder state of size {size} was greater than max batch len: "
             f"{self.split.batch_size}. Prev decoder states producing issue are {prev_decoder_states.shape} and "
             f"split size is {self.split.batch_size.config_option}, and in {type(self).__name__}.")

    def _get_split_size(self, split) -> int:
        return self.split.seq_len.config_option

    def do_register_emit(self, prev_decoder_states, encoder_states, src_mask, tgt_mask,
                         truncate_tensor: bool = False) -> list[(torch.Tensor,torch.Tensor,
                                                                 Optional[torch.Tensor], Optional[torch.Tensor])]:
        registered_value = super().do_register_emit(prev_decoder_states, encoder_states, src_mask, tgt_mask,
                                                    truncate_tensor)
        if self.state_pointer != 0 or self.encoder_state_pointer != 0:
            registered_value.append((
                self.state if prev_decoder_states is not None else None,
                self.encoder_state if encoder_states is not None else None,
                self.src_mask,
                self.tgt_mask
            ))
            self._reset_all_states_and_pointers()
        return registered_value

    def _reset_all_states_and_pointers(self):
        self.state = self._reset_state()
        self.encoder_state = self._reset_state()
        self.src_mask = self._reset_attn()
        self.tgt_mask = self._reset_attn()
        self.tgt_mask_pointer = 0
        self.src_mask_pointer = 0
        self.state_pointer = 0
        self.encoder_state_pointer = 0

    @property
    def split_state_idx(self) -> int:
        return 0

    @property
    def split_attn_idx(self) -> int:
        return 1

    def pad_mask(self, src_mask):
        self._assert_seq_len_size(src_mask, src_mask.size(0) if src_mask is not None else None)
        if src_mask is not None and src_mask.size(0) < self.split.batch_size.config_option:
            s = [self.split.batch_size.config_option, src_mask.size(1)]
            src_mask = pad_add_end_to_match(s, src_mask)
        return src_mask

    def pad_state(self, prev_decoder_states):
        self._assert_seq_len_size(prev_decoder_states,
                                  prev_decoder_states.size(1) if prev_decoder_states is not None else None)
        if prev_decoder_states is not None and prev_decoder_states.size(1) < self.split.batch_size.config_option:
            s = [prev_decoder_states.shape[0], self.split.batch_size.config_option, prev_decoder_states.shape[2]]
            prev_decoder_states = pad_add_end_to_match(s, prev_decoder_states)
        return prev_decoder_states


    def _add_to_state(self, state, prev_decoder_states, batch_pointer):
        new_batch_pointer = batch_pointer + prev_decoder_states.size(self.split_state_idx)

        assert new_batch_pointer // self.split_size <= 2, \
            (f"Batch pointer was {batch_pointer} in state mask. Should be chunked. "
             f"Is {new_batch_pointer // self.split_size}")

        to_add_to = self.permute_to_add_to(prev_decoder_states)

        num_added = min(to_add_to.size(0), self.split_size, min(new_batch_pointer, self.split_size) - batch_pointer)
        state[batch_pointer:min(new_batch_pointer, self.split_size), :, :] \
            = to_add_to[:num_added, :, :]

        if new_batch_pointer > self.split_size:
            out_state = state
            remainder = to_add_to.size(0) - num_added
            state = self._reset_state()
            state[0:remainder, :, :] = to_add_to[to_add_to.size(0) - remainder:, :, :]
        elif new_batch_pointer == self.split_size:
            out_state = state
            state = self._reset_state()
        else:
            out_state = None

        return self.permute_post(out_state), self._mod_batch_ptr(new_batch_pointer), state


    def _add_to_attn_mask(self, mask, curr_state, batch_pointer):
        new_batch_pointer = batch_pointer + mask.size(self.split_attn_idx)

        assert new_batch_pointer // self.split_size <= 2, \
            (f"Batch pointer was {batch_pointer} in attn mask. Should be chunked. "
             f"Is {new_batch_pointer // self.split_size}")

        num_added = min(curr_state.size(self.split_attn_idx), self.split_size, min(new_batch_pointer, self.split_size) - batch_pointer)

        mask = self.permute_attn(mask)
        curr_state = self.permute_attn(curr_state)

        curr_state[:, batch_pointer:min(new_batch_pointer, self.split_size)] = mask[:, :num_added]

        out_state = curr_state

        if new_batch_pointer > self.split_size:
            remainder = mask.size(1) - num_added
            curr_state = self._reset_attn()
            curr_state[:, 0:remainder] = mask[:, mask.size(1) - remainder:]
        elif new_batch_pointer == self.split_size:
            curr_state = self._reset_attn()

        return (self.permute_post_attn(out_state),
                self._mod_batch_ptr(new_batch_pointer),
                self.permute_post_attn(curr_state))

