import torch


def indices_nd(indices_1d, tensor_shape):
    indices_nd = []
    for dim in reversed(tensor_shape):
        indices_nd.append(indices_1d % dim)
        indices_1d = indices_1d // dim

    indices_nd = indices_nd[::1]
    return torch.stack(indices_nd)


def get_top_k_sparse_coo(in_tensor: torch.Tensor, num_k: int, largest: bool=True):
    top_k_out, indices = torch.topk(in_tensor.view(-1), num_k, largest=largest)
    indices = indices_nd(indices, in_tensor.size())
    return torch.sparse_coo_tensor(indices, top_k_out, in_tensor.size())
