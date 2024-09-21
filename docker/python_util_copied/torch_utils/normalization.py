import torch


def do_normalize(in_data: torch.Tensor):
    std_dev = torch.std(in_data, dim=0)
    out = (in_data - torch.mean(in_data, dim=0)) / (std_dev + 1e-9)
    return out


def do_normalize_positive(in_data: torch.Tensor):
    std_dev = torch.std(in_data, dim=0)
    out = torch.sqrt(torch.square(in_data - torch.mean(in_data, dim=0)) / torch.square((std_dev + 1e-9)))
    return out


def do_normalize_subtract(in_data: torch.Tensor):
    return (in_data - torch.mean(in_data)) / torch.mean(in_data)
