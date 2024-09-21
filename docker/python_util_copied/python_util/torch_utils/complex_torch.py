import torch

import numpy as np

from python_util.logger.logger import LoggerFacade
from python_util.torch_utils.pytorch_util import copy_tensor_to


def complex_boltzmann_prob(input_tensor, dim, temperature=1.0):
    """
    Calculate attention scores inspired by quantum mechanics and Boltzmann distribution.

    Args:
        input_tensor (torch.Tensor): Complex-valued input tensor.
        dim (int): Dimension along which the probability scores will be calculated.
        temperature (float, optional): Temperature parameter for the Boltzmann distribution.

    Returns:
        torch.Tensor: Probability scores inspired by quantum mechanics and Boltzmann distribution.
    """
    # Calculate the square of the magnitude of the product of the input tensor and its complex conjugate
    magnitude_squared = torch.abs(input_tensor * input_tensor.conj()) ** 2

    # Exponential term with temperature
    exp_term = torch.exp(magnitude_squared / temperature)

    # Sum along the specified dimension
    prob_scores = exp_term / torch.sum(exp_term, dim=dim, keepdim=True)

    return prob_scores.type(torch.complex64)


def complex_attn_mask(out_mask):
    out_shape = [i for i in out_mask.shape]
    out_shape.append(2)
    expanded_tensor = out_mask.unsqueeze(len(out_shape) - 1).expand(out_shape)
    out_mask = torch.view_as_complex(expanded_tensor.contiguous())
    return out_mask


def ifft_to_real(ifft_taken_already: torch.Tensor):
    magnitude = torch.abs(ifft_taken_already)  # Magnitude of the complex numbers
    phase = torch.angle(ifft_taken_already)  # Phase angle of the complex numbers
    out = (magnitude + 1e-12) * torch.cos(phase)
    return out


def complex_to_2d_2d(complex_numbered: torch.Tensor):
    return torch.view_as_real(complex_numbered)


def complex_to_2d(complex_numbered: torch.Tensor):
    reshaped = torch.view_as_real(complex_numbered)
    if len(reshaped.shape) > 2:
        reshape_size = [i for i in reshaped.shape[:-2]]
        reshape_size.append(reshaped.shape[-1] * reshaped.shape[-2])
        return reshaped.reshape(reshape_size)
    else:
        return reshaped.reshape(reshaped.shape[0] * reshaped.shape[1])


def to_complex_from_2d(to_project_to_complex: torch.Tensor):
    """
    Assumes imaginary part is first
    :param to_project_to_complex:
    :return:
    """
    shape_curr = [s for s in to_project_to_complex.shape[:-1]]
    assert to_project_to_complex.shape[-1] % 2 == 0, "To project to complex must have real and imaginary part."
    shape_curr.extend([to_project_to_complex.shape[-1] // 2, 2])
    to_project_to_complex = to_project_to_complex.reshape(shape_curr)
    return torch.view_as_complex(to_project_to_complex)


def init_complex_weights(in_value, n_frequencies):
    """
    Initializes complex weights with real part in uniform distribution and
    imaginary part in normal distribution with n frequencies.

    Args:
      shape: Desired shape of the complex weight tensor.
      n_frequencies: Number of frequencies for the imaginary part.

    Returns:
      A complex-valued NumPy array with the specified initialization.
    """
    if isinstance(in_value, torch.Size):
        shape = in_value
    elif isinstance(in_value, torch.Tensor):
        shape = in_value.shape
    else:
        LoggerFacade.warn(f"Found in value of type {type(in_value)} when initializing complex weights.")
        if hasattr(in_value, 'shape'):
            shape = in_value.shape
        else:
            shape = in_value

    LoggerFacade.info(f"Initializing complex weights with shape {shape} and num frequency bands {n_frequencies}.")

    # Initialize real part with uniform distribution
    real_part = np.random.uniform(low=-1.0, high=1.0, size=shape)

    # Initialize imaginary part with normal distribution around n frequencies
    freq_bands = np.linspace(1, 1 / np.sqrt(shape[-1]), n_frequencies)
    imag_part = np.zeros(shape)
    for i in range(n_frequencies):
        imag_part += freq_bands[i] * np.random.randn(*shape)

    # Combine real and imaginary parts into a complex-valued array
    complex_weights = real_part + 1j * imag_part

    if not isinstance(in_value, torch.Size) and isinstance(in_value, torch.Tensor):
        copy_tensor_to(torch.tensor(complex_weights), in_value)
        return in_value

    return complex_weights


