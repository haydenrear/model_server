import logging
import typing
from typing import Optional

import torch

from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.padding import pad_add_end_to_match


def merge_masks_into_attn_mask(src_mask: torch.Tensor,
                               tgt_mask: torch.Tensor,
                               n_heads: int):
    # Assume you have source and target key padding masks with the same batch size
    # Extend the masks along the head dimension
    batch_size = src_mask.shape[0]
    src_mask = src_mask.unsqueeze(1).expand(-1, n_heads, -1)
    tgt_mask = tgt_mask.unsqueeze(1).expand(-1, n_heads, -1)

    # Reshape the masks to have a shape of [batch_size * n_heads, ...]
    src_mask = src_mask.reshape(batch_size * n_heads, -1)
    tgt_mask = tgt_mask.reshape(batch_size * n_heads, -1)

    # Use broadcasting to combine the masks
    attn_mask = src_mask.unsqueeze(-1) * tgt_mask.unsqueeze(-2)
    return attn_mask


def merge_masks(multi_head_attn: torch.nn.MultiheadAttention,
                query: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if query is not None:
        query = query.transpose(0, 1)
    if attn_mask is not None and len(attn_mask.shape) == 3 and attn_mask.shape[1] != attn_mask.shape[2]:
        merged = merge_mask_src_tgt(attn_mask, multi_head_attn.num_heads)
        updated = reshape_attn_heads_batch(multi_head_attn, merged)
        return updated
    elif attn_mask is not None and len(attn_mask.shape) == 2:
        new_size = [attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[1]]
        if attn_mask.shape[0] == attn_mask.shape[1]:
            LoggerFacade.warn("Provided 2d attention mask. Assuming that sequence length is same size as batch size. "
                              "If use provided attention mask in format Sequence Length, Sequence Length, this is "
                              f"an error. Resizing to 3d mask of size {new_size}, and then to 4d using torch merge "
                              f"masks.")
        attn_mask = attn_mask.unsqueeze(2).expand(new_size)
    out_value = multi_head_attn.merge_masks(attn_mask=attn_mask,
                                            key_padding_mask=src_key_padding_mask,
                                            query=query)[0]
    if out_value.shape[1] != multi_head_attn.num_heads:
        out_value_reshaped = reshape_attn_heads_batch(multi_head_attn, out_value)
        return out_value_reshaped

    return attn_mask


def make_mask_query_match(mask: Optional[torch.Tensor], query: torch.Tensor):
    assert mask.shape[0] == query.shape[1], "Batch size should match."
    assert len(mask.shape) == 2, "Mask should be provided in form [B, L]"
    if mask is None:
        return get_true_mask_from_input(query)
    elif mask.shape[1] == query.shape[0]:
        return query, mask
    else:
        if mask.shape[1] > query.shape[0]:
            padded = pad_add_end_to_match([
                mask.shape[1], query.shape[1], query.shape[2]
            ], query)
            mask = pad_mask_seq_to_match_query(padded, mask)
            return query, mask
        elif query.shape[0] > mask.shape[1]:
            mask = pad_add_end_to_match([query.shape[1], query.shape[0]], mask)
            return query, mask


def pad_mask_seq_to_match_query(query: torch.Tensor, mask: torch.Tensor):
    if mask.shape[1] > query.shape[0]:
        fill_value = torch.tensor([i for i in range(query.shape[0], query.shape[1])])
        mask.index_fill_(1, fill_value, 0.0)
    return mask


def reshape_attn_heads_batch(multi_head_attn, out_value):
    out_value_shape = [i for i in out_value.shape]
    out_value_shape[1] = multi_head_attn.num_heads
    out_value = out_value.expand(out_value_shape)
    out_value_reshaped = out_value.reshape(
        [out_value_shape[0] * out_value_shape[1], out_value_shape[2], out_value_shape[3]])
    LoggerFacade.debug(f'Reshaped out value to {out_value_reshaped.shape} from {out_value.shape}.')
    return out_value_reshaped


def merge_mask_src_tgt(attn_mask: Optional[torch.Tensor], num_heads: int) -> Optional[torch.Tensor]:
    merged_mask: Optional[torch.Tensor] = None

    if attn_mask is not None:
        # In this branch query can't be a nested tensor, so it has a shape
        batch_size, src_seq_len, tgt_seq_len = attn_mask.shape

        # Always expands attn_mask to 4D
        if attn_mask.dim() == 3:
            attn_mask_expanded = attn_mask.view(batch_size, -1, src_seq_len, tgt_seq_len)
        else:  # attn_mask.dim() == 2:
            attn_mask_expanded = attn_mask.view(1, 1, src_seq_len, tgt_seq_len).expand(batch_size, num_heads, -1, -1)
        merged_mask = attn_mask_expanded

    # no attn_mask and no key_padding_mask, returns None, None
    return merged_mask


def apply_mask_as_value(to_mask: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor],
                        filter_value: float = 0.0,
                        to_set_value: float = 1e-12):
    """
    Applies the value to_set_value to the to_mask at the places where the mask is filter_value.
    :param to_mask:
    :param mask: Accepts mask of dim 2, [Batch Size, Sequence Length]
    :param filter_value:
    :param to_set_value:
    :return:
    """
    mask = mask.transpose(0, 1)
    assert mask.shape[0] == to_mask.shape[0], (f"Mask and to mask must have same batch size, {mask.shape[0]} and "
                                               f"{to_mask.shape[0]} were the batch sizes.")
    assert mask.shape[1] == to_mask.shape[1], (f"Mask and to mask must have same seq len, {mask.shape[1]} and "
                                               f"{to_mask.shape[1]} were the seq len.")
    assert to_mask is not None, ""
    if mask is None:
        return get_true_mask_from_input(to_mask)
    else:
        b = mask
        while len(b.shape) < len(to_mask.shape):
            b = b.unsqueeze(len(b.shape))
        return to_mask.masked_fill_(~b.expand(to_mask.shape).to(dtype=torch.bool), to_set_value)


def get_true_mask_from_input(input_tensor: torch.Tensor):
    shape_value = [i for i in input_tensor.shape]
    assert len(shape_value) == 3, "Input tensor should be of dim 3, where dims are [Length, BatchSize, Features]"
    shape_value.pop()
    shape_value = [shape_value[1], shape_value[0]]
    return torch.ones(shape_value, dtype=torch.float)


def merge_masks_or_threshold(masks_to_merge: list[torch.Tensor], threshold: float):
    """
    Merges masks, where if the number of True as ration of total exceeds threshold, then it is True.
    :param masks_to_merge:
    :param threshold:
    :return:
    """
    assert masks_to_merge
    assert len(masks_to_merge) != 0
    none_size = len(list(filter(lambda x: x is not None, masks_to_merge)))
    assert none_size != 0
    assert none_size == len(masks_to_merge), "None mask provided."
    if len(masks_to_merge) == 1:
        return masks_to_merge[0]

    tester = masks_to_merge[0]
    assert all([len(next_val.shape) == len(tester.shape) for next_val in masks_to_merge]), ("All input masks must have"
                                                                                            "same number of dims.")
    all_same_size = all(
        [torch.allclose(torch.tensor(tester.shape), torch.tensor(next_value.shape)) for next_value in masks_to_merge])
    # assert all_same_size, "All masks must be same size to merge."
    if not all_same_size:
        to_pad_shape = [max([out_val.shape[j] for out_val in masks_to_merge]) for j in range(len(tester.shape))]
        padded_masks_to_merge = [pad_add_end_to_match(to_pad_shape, mask_to_merge) for mask_to_merge
                                 in masks_to_merge]
    else:
        padded_masks_to_merge = masks_to_merge

    threshold_counter = torch.zeros_like(padded_masks_to_merge[0])
    prev = None
    for attn_mask in padded_masks_to_merge:
        if prev is None:
            prev = attn_mask
            continue
        to_threshold = attn_mask.to(torch.bool) & prev.to(torch.bool)
        threshold_counter = to_threshold + threshold_counter.to(torch.float)
        prev = attn_mask

    threshold_counter = threshold_counter.to(torch.float) / (len(padded_masks_to_merge) - 1)
    out_mask = torch.zeros_like(threshold_counter, dtype=torch.float)
    out_mask = out_mask.masked_fill(threshold_counter >= threshold, 1.0)
    return out_mask


def to_float(in_tensor):
    logging.debug(f'{in_tensor.shape if in_tensor is not None else None} is shape.')
    return in_tensor.to(torch.float) if in_tensor is not None else None


def create_padding_mask(input_ids):
    return (input_ids != 0).type(torch.long)


def create_padding_mask_image(input_ids):
    return (input_ids != -1).type(torch.long)


def create_look_ahead_mask(size):
    mask = 1 - torch.triu(torch.ones((1, size, size)), diagonal=1)
    return mask


def create_target_mask(tgt):
    tgt_len = tgt.num_kernels(1)
    tgt_pad_mask = create_padding_mask(tgt)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len)
    tgt_mask = torch.maximum(tgt_pad_mask, tgt_look_ahead_mask)
    return tgt_mask


def make_causal_mask(attn_mask: Optional[torch.Tensor],
                     batch_size: typing.Optional[int] = None,
                     seq_len: typing.Optional[int] = None):
    if attn_mask is None and (batch_size is None or seq_len is None):
        return None
    if len(attn_mask.shape) == 2:
        attn_mask = make_causal_attn_mask_2d(attn_mask, batch_size, seq_len)
    elif len(attn_mask.shape) == 3:
        attn_mask = make_causal_attn_mask_3d(attn_mask)
    return attn_mask


def create_key_padding_mask_from_attn_mask(tgt_mask, batch_size: int):
    if tgt_mask is None:
        return None

    updated_attn_mask = tgt_mask

    if len(tgt_mask.shape) == 2:
        if tgt_mask.shape[0] == tgt_mask.shape[1]:
            updated_attn_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1)
        elif tgt_mask.shape[0] == batch_size:
            return tgt_mask
    elif len(tgt_mask.shape) == 3:
        logging.debug(f'{tgt_mask.shape} is size of output key padding mask from {tgt_mask.shape}')
        assert tgt_mask.shape[0] % batch_size == 0, ("3d attention mask first dimension must be divisible by n heads "
                                                     "size, as it is batch size // n heads.")
        updated_attn_mask = tgt_mask[:tgt_mask.shape[0] // batch_size, 0, :]
    assert len(updated_attn_mask.shape) == 2, (f"Output key padding mask was {updated_attn_mask.shape} "
                                               f"from {updated_attn_mask.shape}")
    return updated_attn_mask


def make_causal_attn_mask_2d(attn_mask: Optional[torch.Tensor],
                             batch_size: Optional[int] = None,
                             seq_len: Optional[int] = None):
    if attn_mask is not None:
        batch_size = attn_mask.size(0)
        seq_len = attn_mask.size(1)
    else:
        attn_mask = torch.ones([batch_size, seq_len])

    # Create a 2D causal mask of size (seq_len, seq_len)
    causal_mask_2d = torch.ones((seq_len, seq_len), dtype=torch.float).tril()

    attn_mask = attn_mask.unsqueeze(1).expand([batch_size, seq_len, seq_len]) * causal_mask_2d

    return attn_mask.to(dtype=torch.float)


def create_random_3d_mask(batch_size, seq_len):
    out = torch.rand([batch_size, seq_len, seq_len])
    out[out > 0.2] = 1.0
    out[out < 0.2] = 0.0
    return out


def create_random_2d_mask(batch_size, seq_len):
    out = torch.rand([batch_size, seq_len])
    out[out > 0.2] = 1.0
    out[out < 0.2] = 0.0
    return out


def make_causal_attn_mask_3d(attn_mask):
    if attn_mask is not None:
        assert len(attn_mask.shape) == 3
        return (torch.ones([i for i in attn_mask.shape], dtype=torch.bool).tril() * attn_mask.type(dtype=torch.bool)).type(torch.float)



def create_attn_mask(n_sequence_state, batch_size, n_heads, decoder_states,
                     attn_mask: Optional[torch.Tensor] = None):
    if attn_mask is not None and len(attn_mask.shape) == 3:
        return create_attn_mask_3d(n_sequence_state, batch_size, n_heads,
                                   decoder_states.shape[0], attn_mask)
    else:
        return create_attn_mask_2d(decoder_states, n_sequence_state, attn_mask)


def create_attn_mask_2d(decoder_states, n_sequence_state,
                        attn_mask: Optional[torch.Tensor] = None):
    padding_size = n_sequence_state - decoder_states.shape[0]
    if attn_mask is None:
        attn_mask = torch.ones((decoder_states.shape[0], decoder_states.shape[0]), dtype=torch.float)
    if padding_size > 0:
        if attn_mask is not None:
            first_dim = n_sequence_state - attn_mask.shape[0]
            second_dim = n_sequence_state - attn_mask.shape[1]
            attn_mask = torch.nn.functional.pad(
                attn_mask,
                pad=(0, min(second_dim, n_sequence_state),
                     0, min(first_dim, n_sequence_state)),
                value=False
            )
    elif attn_mask is not None:
        first_dim = n_sequence_state - attn_mask.shape[0]
        second_dim = n_sequence_state - attn_mask.shape[1]
        attn_mask = torch.nn.functional.pad(
            attn_mask,
            pad=(0, min(second_dim, n_sequence_state),
                 0, min(first_dim, n_sequence_state)),
            value=True
        )
    return attn_mask


def create_attn_mask_3d(n_sequence_state, batch_size, n_heads, seq_len,
                        attn_mask: Optional[torch.Tensor] = None):
    padding_size = n_sequence_state - seq_len
    if attn_mask is None:
        attn_mask = torch.ones((batch_size * n_heads, seq_len, seq_len),
                               dtype=torch.float)
    if padding_size > 0:
        if attn_mask is not None:
            first_dim = batch_size * n_heads - attn_mask.shape[0]
            second_dim = n_sequence_state - attn_mask.shape[1]
            third_dim = n_sequence_state - attn_mask.shape[2]
            attn_mask = torch.nn.functional.pad(
                attn_mask,
                pad=(0, min(third_dim, n_sequence_state),
                     0, min(second_dim, n_sequence_state),
                     0, min(first_dim, batch_size * n_heads)),
                value=False
            )
    elif attn_mask is not None:
        first_dim = batch_size * n_heads - attn_mask.shape[0]
        second_dim = n_sequence_state - attn_mask.shape[1]
        third_dim = n_sequence_state - attn_mask.shape[2]
        attn_mask = torch.nn.functional.pad(
            attn_mask,
            pad=(0, min(third_dim, n_sequence_state),
                 0, min(second_dim, n_sequence_state),
                 0, min(first_dim, batch_size * n_heads)),
            value=True
        )
    return attn_mask


def translate_huggingface_batched_attn_mask_4d(mask_2d: torch.Tensor, num_heads: int):
    N, S = mask_2d.size()

    # Expand the 2D mask to 4D
    mask_4d = mask_2d.unsqueeze(1).unsqueeze(2) * mask_2d.unsqueeze(1).unsqueeze(-1)
    return mask_4d.expand(N, num_heads, S, S)


def translate_huggingface_batched_attn_mask_3d(mask_2d: torch.Tensor):
    N, S = mask_2d.size()
    out = mask_2d.unsqueeze(2).expand(N, S, S)
    return out


def create_key_padding_mask(
        n_sequence_state,
        batch_size,
        prev_decoder_states,
        starting_key_padding_mask: Optional[torch.Tensor] = None
):
    if starting_key_padding_mask is not None:
        assert len(starting_key_padding_mask.shape) == 2, \
            "Key padding mask was expected to be of dimension (batch_size, seq_length)"
    padding_mask = None
    prev_decoder_seq_length = prev_decoder_states.shape[0]
    prev_decoder_batch_size = prev_decoder_states.shape[1]
    if n_sequence_state > prev_decoder_seq_length:
        size_padding = n_sequence_state - prev_decoder_seq_length
        # If max_seq_len < n, pad the tensor
        logging.debug(f'{size_padding} is size of padding')
        if starting_key_padding_mask is not None and len(starting_key_padding_mask.shape) == 1:
            starting_key_padding_mask = starting_key_padding_mask.unsqueeze(0).expand(prev_decoder_batch_size, -1)
        if starting_key_padding_mask is None:
            padding_mask = torch.cat([
                torch.ones(prev_decoder_seq_length, dtype=torch.float),
                torch.zeros(size_padding, dtype=torch.float)
            ])
            padding_mask = padding_mask.unsqueeze(0).expand(prev_decoder_batch_size, -1)
            assert padding_mask.shape[1] == n_sequence_state, \
                "Size of padding mask did not equal sequence state."
        else:
            padding_mask = torch.nn.functional.pad(starting_key_padding_mask,
                                                   pad=(0, n_sequence_state - prev_decoder_states[0], 0, 0))
    if batch_size > prev_decoder_batch_size:
        if padding_mask is not None and batch_size - prev_decoder_batch_size > 0:
            padding_mask = torch.nn.functional.pad(padding_mask,
                                                   pad=(0, 0, 0, batch_size - prev_decoder_batch_size))
        # no padding above - prev decoder state didn't need padding on sequence length, prev decoder length sequence
        # length greater than or equal to n_sequence_state.
        elif padding_mask is None and starting_key_padding_mask is None:
            sequence_padding_mask = torch.ones([prev_decoder_batch_size, prev_decoder_seq_length], dtype=torch.float)
            padding_mask = torch.nn.functional.pad(sequence_padding_mask,
                                                   pad=(0, 0, 0, batch_size - prev_decoder_batch_size))
            assert padding_mask.shape[1] == n_sequence_state, \
                "Size of padding mask did not equal sequence state."
        elif padding_mask is None and starting_key_padding_mask is not None and batch_size > prev_decoder_batch_size:
            padding_mask = torch.nn.functional.pad(starting_key_padding_mask,
                                                   pad=(0, 0, 0, batch_size - starting_key_padding_mask.shape[0]))
    else:
        padding_mask = torch.ones([batch_size, n_sequence_state], dtype=torch.float)
    return padding_mask
