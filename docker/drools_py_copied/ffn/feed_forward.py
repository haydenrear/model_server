import torch

from drools_py.activations.activations import ActivationConfig, ActivationConfigFactory
from drools_py.ffn.ffn_config import FfnConfig, LinearLayerConfig
from python_util.logger.logger import LoggerFacade


class FfnModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, ffn_config: FfnConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = ffn_config.input_dim.config_option
        self.hidden_dim = ffn_config.hidden_dim.config_option
        self.output_dim = ffn_config.output_dim.config_option
        self.layer_norm_first = ffn_config.layer_norm_first.config_option

        self.dtype = ffn_config.dtype.config_option

        self.layer_norm = ffn_config.layer_norm.to_layer_norm()

        self.linear_1 = torch.nn.Linear(self.input_dim, self.hidden_dim, bias=True, dtype=self.dtype)
        self.dropout_one = torch.nn.Dropout(ffn_config.dropout_1.config_option)
        self.activation = ffn_config.activation.create()
        self.linear_2 = torch.nn.Linear(self.hidden_dim, self.output_dim, bias=True, dtype=self.dtype)
        self.dropout_2 = torch.nn.Dropout(ffn_config.dropout_2.config_option)

        if self.layer_norm_first and self.input_dim != self.output_dim:
            LoggerFacade.error(f"Error when initializing FfnModule with {self.input_dim} as input dim and "
                               f"{self.output_dim} as output dim. When using a pointwise connection the input "
                               f"and output dim must be the same!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        if self.layer_norm_first:
            if self.input_dim == self.output_dim:
                return x + self.dropout_2(self.linear_2(self.dropout_one(self.activation(self.linear_1(self.layer_norm(x))))))
            else:
                return self.dropout_2(self.linear_2(self.dropout_one(self.activation(self.linear_1(self.layer_norm(x))))))
        else:
            if self.input_dim == self.output_dim:
                return self.layer_norm(x + self.dropout_2(self.linear_2(self.dropout_one(self.activation(self.linear_1(x))))))
            else:
                return self.layer_norm(self.dropout_2(self.linear_2(self.dropout_one(self.activation(self.linear_1(x))))))


class LinearModule(torch.nn.Module):

    def __init__(self,
                 linear_layer_config: LinearLayerConfig,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer_config = torch.nn.Linear(
            linear_layer_config.in_features.config_option,
            linear_layer_config.out_dim.config_option,
            linear_layer_config.bias.config_option
        )

    def forward(self, input_value):
        return self.linear_layer_config(input_value)