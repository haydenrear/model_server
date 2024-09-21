from typing import Optional

import torch

from python_util.logger.logger import LoggerFacade
from drools_py.positional_encoding.positional_encoding import PositionalEncoding, PositionalEncodingConfig, \
    KayVeePositionalEncoding
from transformers.models.esm.modeling_esm import RotaryEmbedding


class RotaryPositionalEmbedding(KayVeePositionalEncoding):
    def __init__(self,
                 pos_encoding_config: PositionalEncodingConfig,
                 *args, **kwargs):
        super().__init__(pos_encoding_config, *args, **kwargs)
        self.rotary = RotaryEmbedding(pos_encoding_config.d_model.config_option)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q, k = self.rotary(q, k)
        return q.squeeze(0), k.squeeze(0)


class RotaryEmbeddingWithAttention(torch.nn.Module):
    def __init__(self, d_model, max_len, num_heads, num_windows):
        super(RotaryEmbeddingWithAttention, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_windows = num_windows
        self.window_size = max_len // num_windows

        # Rotary embedding factor as learnable parameter
        self.rotary_emb_factor = torch.nn.Parameter(torch.Tensor(1, self.d_model))
        torch.nn.init.uniform_(self.rotary_emb_factor, 0, 0.1)

        # Individual attention for each window
        self.window_attentions = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
            for _ in range(num_windows)
        ])

        # Final attention to aggregate global information
        self.final_attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)

    def forward(self, x):
        # Initial rotary embedding
        pos = torch.arange(self.max_len, dtype=torch.float32, device=x.device).unsqueeze(1)
        rotary_emb = self.rotary_emb_factor * pos / (10000 ** (torch.arange(0, self.d_model, 2, dtype=torch.float32,
                                                                            device=x.device) / self.d_model))
        rotary_emb = torch.cat((torch.cos(rotary_emb), torch.sin(rotary_emb)), dim=-1)
        x += rotary_emb

        # Apply window-wise attention
        for i, window_attention in enumerate(self.window_attentions):
            start_idx = i * self.window_size
            end_idx = (i + 1) * self.window_size
            window_x = x[:, start_idx:end_idx, :]
            attn_output, _ = window_attention(window_x, window_x, window_x)
            x[:, start_idx:end_idx, :] += attn_output

        # Final global attention
        x, _ = self.final_attention(x, x, x)
        return x
