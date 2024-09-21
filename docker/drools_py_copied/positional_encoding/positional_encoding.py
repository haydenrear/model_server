import abc

import numpy as np
import torch.nn.init

from drools_py.configs.config_models import SequenceLength, EmbeddingSize
from drools_py.positional_encoding.positional_encoding_types import PositionalEncodingTypes, \
    PositionalEncodingTypesConfigOption
from drools_py.configs.config import Config
from drools_py.configs.config_factory import ConfigFactory


def srank_func(X):
    # Get the SVD
    (u, s, v) = torch.svd(X)
    sr = (s * s).sum() / s[0] / s[0]
    return sr


def psnr_func(x, y):
    diff = x - y
    err = (diff * diff).flatten().mean().log10()
    return -10 * err


def rff_embedding(x, b=None, embedding_dimension=None):
    if not b:
        b = 5 * torch.randn((int(embedding_dimension / 2), 1))

    return torch.cat((torch.sin((2. * np.pi * x) @ b.T), torch.cos((2. * np.pi * x) @ b.T)), 1)


def basic_encoding(x):
    return torch.cat((torch.sin((2. * np.pi * x)), torch.cos((2. * np.pi * x))), 1)


def rbf_embedding(x, dic, sig):
    return (-0.5 * (x - dic) ** 2 / sig ** 2).exp()


def norm_func(x):
    # Flatten the data
    # x = x.flatten(1,3)

    # Normalize for gain and bias
    y = x - x.mean(1).unsqueeze(-1)
    y = x / x.norm(dim=1).unsqueeze(-1)
    return y


class PositionalEncodingConfig(Config):

    def __init__(self,
                 d_model: EmbeddingSize,
                 max_seq_length: SequenceLength,
                 initialization_type: PositionalEncodingTypes):
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.initialization_type = initialization_type

    @classmethod
    def positional_encoding(cls, d_model: EmbeddingSize, max_seq_len: SequenceLength,
                            initialization_ty: PositionalEncodingTypesConfigOption):
        return PositionalEncodingConfig(d_model, max_seq_len, initialization_ty.config_option)


class PositionalEncodingConfigFactory(ConfigFactory):
    def __init__(self, positional_encoding_config: PositionalEncodingConfig):
        super().__init__(positional_encoding_config)
        self.positional_encoding_config = positional_encoding_config

    def create(self, **kwargs):
        positional_encoding_type = self.positional_encoding_config.initialization_type
        if positional_encoding_type == PositionalEncodingTypes.Cosine:
            from drools_py.positional_encoding.cosine_positional_encoding import CosinePositionalEncoding
            return CosinePositionalEncoding(self.positional_encoding_config)
        if positional_encoding_type == PositionalEncodingTypes.Rotary:
            from drools_py.positional_encoding.rotary_positional_embedding import RotaryPositionalEmbedding
            return RotaryPositionalEmbedding(self.positional_encoding_config)
        else:
            raise ValueError(f"Unknown positional encoding type: {positional_encoding_type}")


class PositionalEncoding(torch.nn.Module):

    def __init__(self, positional_encoding: PositionalEncodingConfig,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positional_encoding = positional_encoding


class KayVeePositionalEncoding(PositionalEncoding):

    @abc.abstractmethod
    def forward(self, q: torch.Tensor, v: torch.Tensor):
        pass
