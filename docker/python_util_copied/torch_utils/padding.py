import torch
import logging

def pad_collapse_states(size_sequence_state: int, to_pad: torch.Tensor, aggregation):
    if len(to_pad.shape) == 3:
        return pad_collapse_states_3d(size_sequence_state, to_pad, aggregation)
    elif len(to_pad.shape) == 2:
        return pad_collapse_states_2d(size_sequence_state, to_pad, aggregation)
    else:
        return to_pad


def fill_with_value_to_match(to_match_size: list[int], to_fill: torch.Tensor, value):
    assert len(to_match_size) == len(to_fill.shape)
    padded = pad_add_end_to_match(to_match_size, to_fill)
    for i, to_match in enumerate(to_match_size):
        if to_match > to_fill.shape[i]:
            index_tensor = torch.tensor([i for i in range(padded.shape[i] - to_match, padded.shape[i])])
            padded.index_fill_(i, index_tensor, value)

    return padded


def pad_collapse_states_2d(size_sequence_state: int, to_pad: torch.Tensor, aggregation):
    # Added option to do_decode_with_encode_if_exists
    if size_sequence_state < to_pad.shape[0]:
        # separate the tensor into two parts
        old_part = to_pad[:size_sequence_state, :]
        new_part = to_pad[size_sequence_state:, :]

        # calculate the mean across the sequence dimension for the old part
        old_part_mean = aggregation.do_agg(old_part, dim=0, keep_dim=True)
        logging.debug(f'{old_part_mean.shape} and {new_part.shape}')

        # concatenate the new part with the old part mean
        logging.debug(f'{to_pad.shape} is previous decoder states before cat')
        to_pad = torch.cat((old_part_mean, new_part[:, :]), dim=0)
        logging.debug(f'{to_pad.shape} is previous decoder states after cat')
    elif size_sequence_state > to_pad.shape[0]:
        # If max_seq_len < n, pad the tensor
        size_padding = size_sequence_state - to_pad.shape[0]
        logging.debug(f'{size_padding} is size of padding')

        padding = (0, 0, 0, size_padding)

        to_pad = torch.nn.functional.pad(to_pad, padding)
        logging.debug(f'{to_pad.shape} is after pad')
    return to_pad


def pad_collapse_states_3d(size_sequence_state: int, to_pad: torch.Tensor, aggregation):
    # Added option to do_decode_with_encode_if_exists
    if size_sequence_state < to_pad.shape[0]:
        # separate the tensor into two parts
        new_part = to_pad[:size_sequence_state, :, :]
        old_part = to_pad[size_sequence_state:, :, :]

        # calculate the mean across the sequence dimension for the old part
        old_part_mean = aggregation.do_agg(old_part, dim=0, keep_dim=True)
        logging.debug(f'{old_part_mean.shape} and {new_part.shape}')

        # concatenate the new part with the old part mean
        logging.debug(f'{to_pad.shape} is previous decoder states before cat')
        to_pad = torch.cat((old_part_mean, new_part[1:, :, :]), dim=0)
        logging.debug(f'{to_pad.shape} is previous decoder states after cat')
    elif size_sequence_state > to_pad.shape[0]:
        # If max_seq_len < n, pad the tensor
        size_padding = size_sequence_state - to_pad.shape[0]
        logging.debug(f'{size_padding} is size of padding')

        padding = (0, 0, 0, 0, 0, size_padding)

        to_pad = torch.nn.functional.pad(to_pad, padding)
        logging.debug(f'{to_pad.shape} is after pad')

    return to_pad


def pad_add_end_to_match(to_match_shape: list[int], to_pad, value = 0.0):
    if all([to_pad.shape[i] == pad for i, pad in enumerate(to_match_shape)]):
        return to_pad
    padding = []
    for dim, size in enumerate(to_match_shape):
        next_padding = max(size - to_pad.shape[dim], 0)
        padding.append(next_padding)
        padding.append(0)

    return torch.nn.functional.pad(to_pad, tuple(list(reversed(padding))), value=value)
