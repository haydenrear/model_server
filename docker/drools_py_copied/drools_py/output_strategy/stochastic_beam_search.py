import torch

from drools_py.output_strategy.output_strategy import OutputStrategy
from drools_py.output_strategy.base_output_strategy_config import OutputStrategyConfig


class StochasticBeamSearch(OutputStrategy):
    def __init__(self, output_strategy_config: OutputStrategyConfig):
        super().__init__(output_strategy_config)
        self.beam_width = output_strategy_config.beam_width
        self.temperature = output_strategy_config.temperature

    def sample(self, logits, sequences):
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)

        # Sample from the probabilities instead of taking the top-k directly
        sampled_indices = torch.multinomial(probabilities, self.beam_width.config_option,
                                            replacement=True)

        # Create new candidates by appending sampled next tokens
        candidates = sequences.unsqueeze(2).expand(-1, -1, self.beam_width.config_option, -1)
        candidates = torch.cat([candidates, sampled_indices.unsqueeze(-1)], dim=-1)

        # Compute new scores for each candidate
        gathered_probabilities = probabilities.gather(-1, sampled_indices)
        candidate_scores = torch.log(gathered_probabilities) + sequences[..., -1, 0].unsqueeze(-1)

        # Flatten the candidates and scores
        candidates = candidates.view(-1, candidates.size(-2), candidates.size(-1))
        candidate_scores = candidate_scores.view(-1, candidate_scores.num_kernels(-1))

        # Select top-k candidates based on the scores
        top_candidate_scores, top_candidate_indices = torch.topk(candidate_scores, self.beam_width.config_option, dim=-1)
        top_candidates = candidates.gather(1, top_candidate_indices.unsqueeze(-1).expand(-1, -1, candidates.num_kernels(-1)))

        return top_candidates, top_candidate_scores

    def forward_layer(self, input_tensor, sequences):
        # This is a placeholder for your model's forward pass
        return torch.rand(input_tensor.num_kernels(0), input_tensor.num_kernels(1), self.beam_width.config_option,
                          self.output_strategy_config.vocab_size.config_option)

    def search(self, initial_input, max_length):
        batch_size = initial_input.num_kernels(0)
        # Initialize sequences and scores
        sequences = torch.full((batch_size, self.beam_width.config_option, 1),
                               self.output_strategy_config.start_token.config_option, dtype=torch.long)
        scores = torch.zeros((batch_size, self.beam_width.config_option), dtype=torch.float)

        # Iteratively expand the beam
        for _ in range(max_length):
            input_tensor = sequences[..., -1, :]
            logits = self.forward_layer(input_tensor, sequences)
            sequences, scores = self.sample(logits, sequences)

        # Return the top sequence from each beam
        return sequences[:, 0, :]
