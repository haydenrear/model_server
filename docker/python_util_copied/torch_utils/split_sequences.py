import logging
from typing import Optional

import torch

from python_util.torch_utils.masking import create_attn_mask_3d, create_attn_mask_2d, create_key_padding_mask


def split_sequence(decoder_states, n_sequence_state) -> list[torch.Tensor]:
    prev_decoder_states: list[torch.Tensor] = []
    num_sequences = decoder_states.shape[0] // n_sequence_state
    num_items_left_over = decoder_states.shape[0] % n_sequence_state
    if num_sequences != 0:
        for i in range(num_sequences):
            if len(decoder_states.shape) == 3:
                prev_decoder_states.append(
                    decoder_states[i * n_sequence_state:i * n_sequence_state + n_sequence_state, :, :])
            elif len(decoder_states.shape) == 2:
                prev_decoder_states.append(
                    decoder_states[i * n_sequence_state:i * n_sequence_state + n_sequence_state, :])

    if num_items_left_over != 0:
        end_start = num_sequences * n_sequence_state

        if len(decoder_states.shape) == 2:
            decoder_state = decoder_states[end_start:end_start + num_items_left_over, :]
            # decoder_state = torch.nn.functional.pad(decoder_state,
            #                                         pad=(0, 0, 0, n_sequence_state - num_items_left_over))
            prev_decoder_states.append(decoder_state)
        if len(decoder_states.shape) == 3:
            decoder_state = decoder_states[end_start:end_start + num_items_left_over, :, :]
            # decoder_state = torch.nn.functional.pad(decoder_state,
            #                                         pad=(0, 0, 0, 0, 0, n_sequence_state - num_items_left_over))
            prev_decoder_states.append(decoder_state)

    return prev_decoder_states


def split_attn_mask_2d(attn_mask, n_sequence_state):
    attn_masks = []
    num_sequences = attn_mask.shape[0] // n_sequence_state
    num_items_left_over = attn_mask.shape[0] % n_sequence_state
    if num_sequences != 0:
        logging.debug(f'{num_sequences} is num sequences when splitting 2d attn mask.')
        for i in range(num_sequences):
            attn_masks.append(attn_mask[i * n_sequence_state:i * n_sequence_state + n_sequence_state,
                              i * n_sequence_state:i * n_sequence_state + n_sequence_state])

    if num_items_left_over != 0:
        end_start = num_sequences * n_sequence_state
        starting = attn_mask[end_start:end_start + num_items_left_over, end_start:end_start + num_items_left_over]
        starting = torch.nn.functional.pad(starting, pad=(
            0, n_sequence_state - num_items_left_over, 0, n_sequence_state - num_items_left_over), value=False)
        attn_masks.append(starting.to(torch.float) if starting is not None else None)

    return [attn_mask.to(torch.float) for attn_mask in attn_masks]


def split_key_padding_mask(key_padding_mask, n_sequence_state):
    if key_padding_mask is None:
        return []
    attn_masks = []
    num_sequences = key_padding_mask.shape[1] // n_sequence_state
    num_items_left_over = key_padding_mask.shape[1] % n_sequence_state
    if num_sequences != 0:
        for i in range(num_sequences):
            attn_masks.append(key_padding_mask[:, i * n_sequence_state:i * n_sequence_state + n_sequence_state])

    if num_items_left_over != 0:
        end_start = num_sequences * n_sequence_state
        starting = key_padding_mask[:, end_start: end_start + num_items_left_over]
        starting = torch.nn.functional.pad(starting, pad=(0, n_sequence_state - num_items_left_over, 0, 0), value=False)
        attn_masks.append(starting)

    return [attn_mask.to(torch.float) for attn_mask in attn_masks]


def split_attn_mask_3d(attn_mask, n_sequence_state):
    attn_masks = []
    num_sequences = attn_mask.shape[1] // n_sequence_state
    num_items_left_over = attn_mask.shape[1] % n_sequence_state
    if num_sequences != 0:
        for i in range(num_sequences):
            attn_masks.append(attn_mask[:, i * n_sequence_state:i * n_sequence_state + n_sequence_state,
                              i * n_sequence_state:i * n_sequence_state + n_sequence_state])

    if num_items_left_over != 0:
        end_start = num_sequences * n_sequence_state
        attn_masks.append(
            attn_mask[:, end_start:end_start + num_items_left_over, end_start:end_start + num_items_left_over])

    return [attn_mask.to(torch.float) for attn_mask in attn_masks]


def get_split_attn_masks(
        decoder_states,
        n_sequence_state,
        n_attn_heads,
        attn_mask: Optional[torch.Tensor] = None):
    to_add = n_sequence_state - (decoder_states.shape[0] % n_sequence_state)
    intermediary_sequence_state = decoder_states.shape[0] + to_add // n_sequence_state
    if attn_mask is not None:
        if len(decoder_states.shape) == 3 and len(attn_mask.shape) == 3:
            assert_attn_mask_seq_length_matches(attn_mask)
            if attn_mask.shape[0] != decoder_states.shape[1] * n_attn_heads:
                raise_attn_mask_n_heads_error(attn_mask, decoder_states, n_attn_heads)
            if attn_mask.shape[1] < decoder_states.shape[0]:
                attn_mask = create_attn_mask_3d(intermediary_sequence_state, decoder_states.shape[0], n_attn_heads,
                                                decoder_states.shape[0], attn_mask)
            elif attn_mask.shape[1] != decoder_states.shape[0]:
                raise_attn_mask_shape_error(attn_mask, decoder_states)
        elif len(decoder_states.shape) == 2 and len(attn_mask.shape) == 3:
            assert_attn_mask_seq_length_matches(attn_mask)
            if attn_mask.shape[0] != n_attn_heads:
                raise_attn_mask_n_heads_error(attn_mask, decoder_states, n_attn_heads)
            if attn_mask.shape[1] < decoder_states.shape[0]:
                attn_mask = create_attn_mask_3d(intermediary_sequence_state, decoder_states.shape[0], n_attn_heads,
                                                decoder_states.shape[0], attn_mask)
            elif attn_mask.shape[1] != decoder_states.shape[0]:
                raise_attn_mask_shape_error(attn_mask, decoder_states)
        elif len(attn_mask.shape) == 2:
            assert attn_mask.shape[0] == attn_mask.shape[1], \
                f"Attention mask shape did not match on sequence length: {attn_mask.shape}."
            if attn_mask.shape[0] < decoder_states.shape[0]:
                attn_mask = create_attn_mask_2d(decoder_states, intermediary_sequence_state, attn_mask)
            elif attn_mask.shape[0] != decoder_states.shape[0]:
                raise_attn_mask_shape_error(attn_mask, decoder_states)

        attn_masks = split_attn_mask(attn_mask, n_sequence_state)
    else:
        if intermediary_sequence_state > decoder_states.shape[0]:
            starting = torch.ones([decoder_states.shape[0], decoder_states.shape[0]], dtype=torch.float)
            padding = torch.zeros([intermediary_sequence_state - decoder_states.shape[0],
                                   intermediary_sequence_state - decoder_states.shape[0]], dtype=torch.float)
            mask = torch.cat([starting, padding])
        else:
            assert intermediary_sequence_state == decoder_states.shape[0], \
                "Multiple of sequence size was not greater than decoder sequence length."
            mask = torch.ones([intermediary_sequence_state, intermediary_sequence_state], dtype=torch.float)
        attn_masks = split_attn_mask(mask, n_sequence_state)

    return [torch.nn.functional.pad(attn_m, pad=(0, n_sequence_state - attn_m.shape[1],
                                                 0, n_sequence_state - attn_m.shape[2],
                                                 0, 0))
            if len(attn_m.shape) == 3 else
            torch.nn.functional.pad(attn_m, pad=(0, n_sequence_state - attn_m.shape[1],
                                                 0, 0))
            for attn_m in attn_masks]


def get_key_padding_mask(
        decoder_states,
        n_sequence_state,
        batch_size: Optional[int] = 1,
        key_padding_mask: Optional[torch.Tensor] = None):
    if key_padding_mask is not None:

        to_add = n_sequence_state - (decoder_states.shape[0] % n_sequence_state)
        intermediary_sequence_state = decoder_states.shape[0] + to_add // n_sequence_state

        if len(decoder_states.shape) == 3:
            if key_padding_mask.shape[1] < decoder_states.shape[0]:
                key_padding_mask = create_key_padding_mask(
                    intermediary_sequence_state,
                    batch_size,
                    decoder_states,
                    key_padding_mask
                )
            elif key_padding_mask.shape[1] != decoder_states.shape[0]:
                raise ValueError(f"Key padding mask sequence length of shape {key_padding_mask.shape[1]} did not "
                                 f"match sequence length of {decoder_states.shape[0]} for key padding mask shape of "
                                 f"{key_padding_mask.shape} and sequence shape of {decoder_states.shape} and was not "
                                 f"less than sequence shape, was greater than.")

        return split_key_padding_mask(key_padding_mask, n_sequence_state)
    else:
        mask = torch.ones(
            [decoder_states.shape[1] if len(decoder_states.shape) == 3 else batch_size, decoder_states.shape[0]],
            dtype=torch.float)
        return split_key_padding_mask(mask, n_sequence_state)


def split_attn_mask(attn_mask, n_sequence_state):
    if len(attn_mask.shape) == 3:
        attn_mask = split_attn_mask_3d(attn_mask, n_sequence_state)
    elif len(attn_mask.shape) == 2:
        attn_mask = split_attn_mask_2d(attn_mask, n_sequence_state)
    return [attn_mask_item.to(torch.float) for attn_mask_item in attn_mask]


def raise_attn_mask_n_heads_error(attn_mask, decoder_states, n_attn_heads):
    raise ValueError(f"{attn_mask.shape[0]} batch size * n_attn_heads "
                     f"did not match {decoder_states.shape[1] * n_attn_heads}. 3d attention padding "
                     f"mask: {attn_mask.shape} did not match batch size of sequence {decoder_states.shape}.")


def raise_attn_mask_shape_error(attn_mask, decoder_states):
    raise ValueError(
        f"Attention mask sequence length {attn_mask.shape[1]} was greater than decoder state sequence "
        f"length {decoder_states.shape[0]}. 3d attention padding "
        f"mask: {attn_mask.shape} did not match batch size of sequence {decoder_states.shape}.")


def assert_attn_mask_seq_length_matches(attn_mask):
    assert attn_mask.shape[1] == attn_mask.shape[2], \
        f"Attention mask shape did not match on sequence length: {attn_mask.shape}."


def split_sequence_get_attn_mask(
        decoder_states,
        n_sequence_state,
        n_attn_heads: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None
) -> list[(torch.Tensor, Optional[torch.Tensor])]:
    if n_attn_heads is None and len(decoder_states.shape) == 3 and decoder_states.shape[1] != 1:
        assert attn_mask is not None, "Need number of attention heads for 3d attention mask."

    split_states = split_sequence(decoder_states, n_sequence_state)
    if attn_mask is None:
        return [(decoder, None) for decoder in split_states]
    else:
        masks = get_split_attn_masks(decoder_states, n_sequence_state, n_attn_heads, attn_mask)
        assert len(split_states) == len(masks), \
            "If attention mask exists, the size of the output of the splitting of the state and the attention mask " \
            f"must be equal. {len(split_states)} was the size of the state and {len(masks)} was the size of the mask"
        return [(split_states[i], masks[i]) for i in range(len(split_states))]


def split_sequence_get_key_padding_mask(
        decoder_states,
        n_sequence_state,
        key_padding_mask: Optional[torch.Tensor] = None
) -> list[(torch.Tensor, Optional[torch.Tensor])]:
    split_states = split_sequence(decoder_states, n_sequence_state)
    if key_padding_mask is None:
        return [(decoder, None) for decoder in split_states]
    else:
        masks = get_key_padding_mask(decoder_states, n_sequence_state, key_padding_mask=key_padding_mask)
        assert len(split_states) == len(masks), \
            "If attention mask exists, the size of the output of the splitting of the state and the attention mask " \
            f"must be equal. {len(split_states)} was the size of the state and {len(masks)} was the size of the mask"

        return [(split_states[i], masks[i]) for i in range(len(split_states))]


def split_seq(to_split, seq_state, chunk_inputs):
    if chunk_inputs:
        seq_state_found = seq_state
    else:
        seq_state_found = to_split.shape[0]

    split_seq_ = split_sequence(to_split, seq_state_found)

    return split_seq_


def split_sequences_key_query_value_decoder(key, query, value, decoder_input, seq_state, chunk_inputs):
    key_sequences = split_seq(key, seq_state, chunk_inputs)
    query_sequences = split_seq(query, seq_state, chunk_inputs)
    if value is None and key is not None:
        value_sequences = key_sequences
    elif value is None:
        assert query is not None, "Both query, key, and value were not provided."
        value_sequences = query_sequences
    else:
        value_sequences = split_seq(value, seq_state, chunk_inputs)
    if decoder_input is None and key is not None:
        decoder_input_sequences = key_sequences
    elif decoder_input is None:
        assert query is not None, "Both query, key, and value were not provided."
        decoder_input_sequences = query_sequences
    else:
        decoder_input_sequences = split_seq(decoder_input, seq_state, chunk_inputs)

    return key_sequences, query_sequences, value_sequences, decoder_input_sequences


def get_split_sequences_masks(tgt_mask: torch.Tensor,
                              sequences: list[torch.Tensor]):
    idx_attn_mask = 0
    masks = []
    for i in range(len(sequences)):
        next_src_mask = tgt_mask[:, idx_attn_mask:idx_attn_mask + sequences[i].shape[0]]
        if not torch.is_floating_point(next_src_mask):
            next_src_mask = next_src_mask.to(dtype=torch.float)
        masks.append(next_src_mask)
        idx_attn_mask += sequences[i].shape[0]

    return masks
