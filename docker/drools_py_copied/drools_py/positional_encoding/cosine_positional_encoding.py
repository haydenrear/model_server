import math

import torch.nn

from drools_py.positional_encoding.positional_encoding import PositionalEncodingConfig, PositionalEncoding


class CosinePositionalEncoding(PositionalEncoding):
    def __init__(self,
                 pos_encoding_config: PositionalEncodingConfig,
                 *args, **kwargs):
        super().__init__(pos_encoding_config, *args, **kwargs)
        self.pos_encoding_config = pos_encoding_config
        self.d_model = self.pos_encoding_config.d_model

        pe = torch.zeros(pos_encoding_config.max_seq_length.config_option, self.d_model.config_option)

        position = torch.arange(0, self.pos_encoding_config.max_seq_length.config_option, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model.config_option, 2)).float() * (-math.log(10000.0) / self.d_model.config_option)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe = torch.nan_to_num(pe, 0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length, batch_size, features = x.size()
        return x + self.pe[:seq_length, :batch_size, :features]
