import torch.nn

from drools_py.ffn.feed_forward import FfnModule
from drools_py.ffn.ffn_config import DenseFfnConfig


class DenseFfn(torch.nn.Module):
    def __init__(self, dense_net_config: DenseFfnConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_net_config = dense_net_config
        self.intermediate = torch.nn.Sequential(*[
            FfnModule(config) for config
            in dense_net_config.intermedia_layers
        ])
        self.out = FfnModule(dense_net_config.out_layer)

    def forward(self, in_data: torch.Tensor):

        outputs = [in_data]

        for next_layer in self.intermediate:
            next_value = torch.cat(outputs, dim=-1)
            next_value = next_layer(next_value)
            outputs.append(next_value)

        return self.out(torch.cat(outputs, dim=-1))

