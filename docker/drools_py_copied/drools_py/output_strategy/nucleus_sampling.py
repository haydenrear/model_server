import torch

from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig


class NucleusSampling(OutputStrategy):
    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)
        self.p = output_strategy_config.top_p

    def sample(self, logits):
        assert 0 < self.p <= 1

        sorted_value, sorted_index = torch.sort(logits, dim=-1, descending=True)

        sofmax_tensor = torch.softmax(sorted_value, dim=-1)
        cumsum_tensor = torch.cumsum(sofmax_tensor, dim=-1)
        masked_tensor = cumsum_tensor > self.p
        assert isinstance(masked_tensor, torch.Tensor), "Mask was not created successfully."

        sorted_logits = torch.masked_scatter(sorted_value, masked_tensor,
                                             torch.full(sorted_value.shape, -1e13, device=logits.device))
        sorted_output = torch.multinomial(torch.softmax(sorted_logits, dim=-1),
                                          num_samples=self.output_strategy_config.num_samples)
        output = torch.gather(sorted_index, dim=1, index=sorted_output).view(-1)

        return output
