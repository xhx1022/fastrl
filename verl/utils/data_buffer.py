import logging
from collections import deque
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DataBuffer:
    """Buffer to store training data from multiple RL steps for draft model training.

    This buffer accumulates data (input_ids, responses, prompts, hidden_states) across
    RL training steps, allowing the draft model to train on a larger dataset that includes
    both current and previous step data.

    Args:
        max_size: Maximum number of samples to store in buffer
        store_hidden_states: Whether to store hidden_states (default: True)
    """

    def __init__(self, max_size: int = 10000, store_hidden_states: bool = True):
        self.max_size = max_size
        self.store_hidden_states = store_hidden_states
        self.buffer = deque(maxlen=max_size)
        self._current_step = 0

    def add_batch(self, batch: dict[str, torch.Tensor], hidden_states: Optional[list[torch.Tensor]] = None):
        """Add a batch of data to the buffer.

        Args:
            batch: Dictionary containing:
                - input_ids: Tensor of shape [batch_size, seq_len]
                - responses: Tensor of shape [batch_size, response_len]
                - prompts: Tensor of shape [batch_size, prompt_len]
            hidden_states: List of tensors, one per sample in batch.
                Each tensor has shape [seq_len, hidden_dim] or [1, seq_len, hidden_dim]
        """
        input_ids = batch.get("input_ids")
        responses = batch.get("responses")
        prompts = batch.get("prompts")

        if input_ids is None:
            logger.warning("Cannot add batch without input_ids")
            return

        batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1

        # Add each sample individually to the buffer
        for i in range(batch_size):
            sample = {
                "input_ids": input_ids[i].detach().cpu() if input_ids.dim() > 1 else input_ids.detach().cpu(),
                "step": self._current_step,
            }

            if responses is not None:
                sample["responses"] = responses[i].detach().cpu() if responses.dim() > 1 else responses.detach().cpu()

            if prompts is not None:
                sample["prompts"] = prompts[i].detach().cpu() if prompts.dim() > 1 else prompts.detach().cpu()

            if self.store_hidden_states and hidden_states is not None and i < len(hidden_states):
                h_state = hidden_states[i]
                # Handle different tensor shapes
                if h_state.dim() == 3:
                    h_state = h_state.squeeze(0)  # Remove batch dimension if present
                elif h_state.dim() == 1:
                    h_state = h_state.unsqueeze(0)  # Add seq_len dimension
                sample["hidden_states"] = h_state.detach().cpu()

            self.buffer.append(sample)

    def increment_step(self):
        """Increment the current RL step counter."""
        self._current_step += 1

    def get_all_data(self) -> list[dict[str, torch.Tensor]]:
        """Get all data from the buffer.

        Returns:
            List of dictionaries, each containing data for one sample
        """
        return list(self.buffer)

    def get_data_from_last_n_steps(self, n: int) -> list[dict[str, torch.Tensor]]:
        """Get data from the last n RL steps.

        Args:
            n: Number of recent steps to retrieve data from

        Returns:
            List of dictionaries containing data from last n steps
        """
        min_step = max(0, self._current_step - n)
        return [sample for sample in self.buffer if sample["step"] >= min_step]

    def get_data_count(self) -> int:
        """Get the current number of samples in the buffer."""
        return len(self.buffer)

    def get_data_count_from_last_n_steps(self, n: int) -> int:
        """Get number of samples from the last n steps."""
        return len(self.get_data_from_last_n_steps(n))

    def clear(self):
        """Clear all data from the buffer."""
        self.buffer.clear()
        self._current_step = 0

    def get_current_step(self) -> int:
        """Get the current RL step number."""
        return self._current_step

    def __len__(self) -> int:
        """Return the number of samples in buffer."""
        return len(self.buffer)
