import torch

from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig


class TopKSampling(OutputStrategy):
    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)
        self.k = output_strategy_config.top_k

    def sample(self, logits):
        assert self.k > 0

        sorted_value, sorted_index = torch.sort(logits, dim=-1, descending=True)

        arange_tensor = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0).repeat(logits.shape[0], 1)
        kvalue_tensor = torch.tensor(logits.shape[1] * [self.k], device=logits.device).unsqueeze(0).repeat(
            logits.shape[0], 1)
        masked_tensor = arange_tensor >= kvalue_tensor

        assert isinstance(masked_tensor, torch.Tensor), "Mask not created successfully."

        sorted_logits = torch.masked_scatter(sorted_value, masked_tensor,
                                             torch.full(sorted_value.shape, -1e13, device=logits.device))
        sorted_output = torch.multinomial(torch.softmax(sorted_logits, dim=-1), num_samples=1)
        output = torch.gather(sorted_index, dim=1, index=sorted_output).view(-1)

        return output
