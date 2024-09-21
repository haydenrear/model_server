import typing
from typing import Optional

import torch.nn
import numpy

from drools_py.mask.attn_mask_delegate import AttnMaskDelegate
from python_util.logger.log_level import LogLevel
from python_util.logger.logger import LoggerFacade


def reshape_frontload(outputs):
    if len(outputs.shape) == 1:
        return outputs
    if len(outputs.shape) == 2:
        return outputs.reshape(outputs.shape[0] * outputs.shape[1])
    start = 1
    for o in outputs.shape[:-1]:
        start *= o
    reshape_output = [start, outputs.shape[-1]]
    return outputs.reshape(reshape_output)


def assert_same(t_1, t_2):
    assert torch.allclose(t_1, t_2)


def log_attn_mask(in_torch: AttnMaskDelegate):
    if in_torch is not None:
        return ', '.join([f'{create_torch_size_log(in_torch.src_mask)}',
                          f'{create_torch_size_log(in_torch.tgt_mask)}',
                          f'{create_torch_size_log(in_torch.merged_mask)}',
                          f'{create_torch_size_log(in_torch.causal_mask)}',])
    else:
        return 'Empty attention mask.'


def create_torch_size_log(in_torch: Optional[torch.Tensor]):
    try:
        return (
            f"Shape: [{', '.join([str(i) for i in (in_torch if isinstance(in_torch, torch.Size) else in_torch.shape)])}], "
            f"dtype: {in_torch.data.dtype}.") \
            if in_torch is not None else None
    except Exception as e:
        return f"Could not return shape for {in_torch} with exception {e}."


def create_torch_grad_log(in_torch: Optional[torch.Tensor]):
    try:
        return (f"Grad: [{in_torch.grad}], "
                f"Grad fn: {in_torch.grad_fn}.") \
            if in_torch is not None else None
    except Exception as e:
        return f"Could not return shape for {in_torch} with exception {e}."


def is_same_shape(t_1, t_2):
    t_1 = get_torch_shape(t_1)
    t_2 = get_torch_shape(t_2)
    same_shape = torch.allclose(t_1.to(dtype=torch.float), t_2.to(dtype=torch.float))
    if not same_shape:
        LoggerFacade.error(f"Torch sizes were not same shape. {t_1} compared to {t_2}.")

    return same_shape


def assert_same_shape(t_1, t_2):
    assert is_same_shape(t_1, t_2), f"{t_1} not same as {t_2}."


def get_torch_shape(t_1):
    if isinstance(t_1, typing.List) or isinstance(t_1, typing.Iterable) or isinstance(t_1, typing.Iterator):
        t_1 = torch.tensor(t_1)
    elif isinstance(t_1, torch.Tensor):
        t_1 = torch.tensor(t_1.shape)
    elif isinstance(t_1, torch.Size):
        t_1 = torch.tensor(t_1)
    return t_1


def shift_attn_mask_start(input_mask: torch.Tensor, num_shift: int):
    batch_size = input_mask.size(0)
    ones_column = torch.ones([batch_size, num_shift], dtype=torch.float)
    shifted_mask = torch.cat([ones_column, input_mask[:, num_shift:]], dim=1)
    return shifted_mask


def shift_attn_mask_end(input_mask: torch.Tensor, num_shift: int):
    batch_size = input_mask.size(0)
    ones_column = torch.zeros([batch_size, num_shift], dtype=torch.float)
    shifted_mask = torch.cat([input_mask[:, num_shift:], ones_column], dim=1)
    return shifted_mask


def shift_seq_right_end(input_seq: torch.Tensor, num_shift: int, shift_token: Optional = None):
    if shift_token is None:
        return input_seq[num_shift:, :, :]
    if isinstance(shift_token, float | int):
        return torch.nn.functional.pad(input_seq[num_shift:, :, :], (0, 0, 0, 0, 0, num_shift), value=shift_token)
    elif isinstance(shift_token, torch.Tensor):
        return shift_seq_right_vec_end(input_seq, num_shift, shift_token)


def shift_input_ids_right_start(input_seq: torch.Tensor, num_shift: int, start_token_id):
    if isinstance(start_token_id, int):
        return torch.nn.functional.pad(input_seq[:, :-num_shift], (num_shift + 1, 0, 0, 0),
                                       value=start_token_id)
    elif isinstance(start_token_id, torch.Tensor):
        raise ValueError("Start token id cannot be tensor.")


def shift_input_ids_right_end(input_seq: torch.Tensor, num_shift: int, end_token_id):
    if isinstance(end_token_id, int):
        return torch.nn.functional.pad(input_seq[:, num_shift:], (0, num_shift, 0, 0), value=end_token_id)
    elif isinstance(end_token_id, torch.Tensor):
        raise ValueError("Start token id cannot be tensor.")


def shift_seq_right_start(input_seq: torch.Tensor, num_shift: int, start_token_id: Optional = None):
    if isinstance(start_token_id, int | float):
        return torch.nn.functional.pad(input_seq[:-num_shift, :, :], (0, 0, 0, 0, num_shift + 1, 0),
                                       value=start_token_id)
    elif isinstance(start_token_id, torch.Tensor):
        return shift_seq_right_vec_start(input_seq, num_shift, start_token_id)


def shift_seq_right_vec_start(input_seq: torch.Tensor, num_shift: int, shift_vec):
    shift_vec = shift_vec.expand([1, input_seq.shape[1], shift_vec.shape[0]])
    return torch.cat([shift_vec, input_seq[:-num_shift, :, :], ], dim=0)


def shift_seq_right_vec_end(input_seq: torch.Tensor, num_shift: int, shift_vec):
    shift_vec = shift_vec.expand([1, input_seq.shape[1], shift_vec.shape[0]])
    return torch.cat([input_seq[num_shift:, :, :], shift_vec], dim=0)


def split_tensor(input_tensor: torch.Tensor, embedding_size: int) -> (dict[int, torch.Tensor],
                                                                      dict[int, torch.Tensor]):
    split_tensors = torch.split(input_tensor, embedding_size)
    split_dict = {i: split_tensors[i] for i in range(len(split_tensors))}

    # Create a dictionary to store the indices.
    index_dict = {i: (i * embedding_size, min((i + 1) * embedding_size, input_tensor.shape[0]))
                  for i in range(len(split_tensors))}

    return split_dict, index_dict


def reconstruct_tensor(split_dict: dict[int, torch.Tensor], index_dict: dict[int, torch.Tensor]):
    # Initialize a list to hold the reassembled tensor parts
    tensor_parts = []

    # Iterate over the index_dict items, which are sorted by their keys
    for i, (start, end) in sorted(index_dict.items()):
        # Append the corresponding tensor from split_dict to tensor_parts
        tensor_parts.append(split_dict[i])

    # Concatenate all tensor parts along the first dimension to get the original tensor
    reconstructed_tensor = torch.cat(tensor_parts)

    return reconstructed_tensor


def insert_at_indices_no_proj(tensor, indices, value):
    """
    :param tensor:
    :param indices: The indices in coordinate space.
    :param value:
    :return:
    """
    new_tensor_size = list(tensor.shape)
    new_tensor_size[0] += indices.shape[0]
    if not isinstance(value, torch.Tensor):
        new_tensor = torch.full(new_tensor_size, fill_value=value, dtype=tensor.dtype)
    else:
        new_tensor = torch.full(new_tensor_size, fill_value=0, dtype=tensor.dtype)
        for i in range(indices.shape[0]):
            for j, index in enumerate(indices[i]):
                new_tensor[tuple(index)] = value[i][j]

    mask = torch.ones(new_tensor_size, dtype=torch.bool)
    for i in range(indices.shape[0]):
        for index in indices[i]:
            index = index.numpy().tolist()
            mask[tuple(index)] = 0

    flattened = new_tensor.flatten()
    flattened[mask.flatten()] = tensor.flatten()

    return flattened.reshape(*new_tensor_size)


def insert_at_indices(tensor, indices, value):
    """
    :param tensor:
    :param indices: The indices, not in coordinate space,
    :param value:
    :return:
    """
    new_tensor_size = list(tensor.shape)
    new_tensor_size[0] += len(indices)
    if not isinstance(value, torch.Tensor):
        new_tensor = torch.full(new_tensor_size, fill_value=value, dtype=tensor.dtype)
    else:
        new_tensor = torch.full(new_tensor_size, fill_value=0, dtype=tensor.dtype)
        num_values_adding = torch.prod(torch.tensor([i for i in new_tensor.shape])) \
                            - torch.prod(torch.tensor([i for i in tensor.shape]))
        num_values_in_values = torch.prod(torch.tensor([val for val in value.shape]))
        if num_values_in_values != num_values_adding:
            raise ValueError(f"Failed because number of values in values tensor was different: {num_values_adding} "
                             f"did not match {num_values_in_values}")
        try:
            for i in range(value.shape[0]):
                index = tuple(indices[i])  # Convert to tuple
                new_tensor[index] = value[i]
        except:
            for i in range(value.shape[0]):
                new_tensor[indices[i]] = value[i]

    mask = torch.ones(new_tensor_size[0], dtype=torch.bool)
    mask[indices] = 0

    new_tensor[mask] = tensor

    return new_tensor


def drop_indices(tensor, indices):
    mask = torch.ones(tensor.num_kernels(0), dtype=torch.bool)
    mask[indices] = 0
    return tensor[mask]


def get_batch_size(decoder_states):
    batch_size = decoder_states.shape[1] if len(decoder_states.shape) == 3 else 1
    return batch_size


def get_feature_size(state):
    if state is None:
        return 0
    assert len(state.shape) == 3 or len(state.shape) == 2, \
        "Tensor should be either 2 or 3 dimensions."
    if len(state.shape) == 3:
        return state.shape[2]
    elif len(state.shape) == 2:
        return state.shape[1]


def get_sequence_size(state):
    if state is None:
        return 0
    assert len(state.shape) == 3 or len(state.shape) == 2, \
        "Tensor should be either 2 or 3 dimensions."
    return state.shape[0]


def does_tensor_have_nan(in_tensor: torch.Tensor) -> bool:
    return numpy.isnan(in_tensor.detach().numpy()).any()


def copy_tensor_to(copy_to: torch.Tensor, copy_from: torch.Tensor):
    if copy_to.data.dtype != copy_from.data.dtype:
        copy_to.to(dtype=copy_from.data.dtype)
    copy_to.data = copy_from.data.clone().detach()
    copy_to.requires_grad_(copy_from.requires_grad)


def do_nan(in_tensor: torch.Tensor, value: float = 0.0, do_if: bool = False) -> torch.Tensor:
    if LogLevel.is_debug_enabled() and not do_if:
        assert not does_tensor_have_nan(in_tensor), f"{in_tensor} had NAN."
    else:
        if do_if and LogLevel.is_debug_enabled():
            LoggerFacade.debug(f"Updating {in_tensor} to fix NAN.")
        in_tensor = torch.nan_to_num(in_tensor, value)
    return in_tensor


def detach_tensor_state(input_state):
    input_state.prev_decoder_state \
        = input_state.data.clone().detach().requires_grad_(False)
    return input_state
