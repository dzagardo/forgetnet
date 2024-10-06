# forgetnet/dp/dp_shuffle.py

import math
from typing import List, Tuple
import torch
from ..core import PrivacyMechanism
from transformers import modeling_utils
import torch.nn as nn
from typing import Dict, Any

class DPShuffleGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        target_epsilon: float,
        delta: float,
        steps: int,
        base_clip_value: float,
        batch_size: int,
        dataset_size: int,
        eta: float = 1e-6,
        clip_factor: float = 0.9
    ):
        self.model = model
        self.dataset_size = dataset_size
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps = steps
        self.base_clip_value = base_clip_value
        self.batch_size = batch_size
        self.clip_factor = clip_factor
        self.accountant = DPShufflePrivacyAccountant(
            model,
            target_epsilon,
            # delta,
            steps,
            base_clip_value,
            batch_size,
            dataset_size,
            eta
        )
        self.optimal_block_sizes = self.accountant.optimize_parameters()
        print(f"Optimal block sizes: {self.optimal_block_sizes}")
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
        self.current_step = 0
        self.adaptive_clip_values = {}

    def apply(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        private_grads, _, _ = self.generate(gradients)
        return private_grads

    def generate(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float, float]:
        private_grads = []
        grad_index = 0
        self.adaptive_clip_values = {}

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module) and list(module.parameters()):
                module_grads = []
                for _ in module.parameters():
                    if grad_index < len(gradients):
                        module_grads.append(gradients[grad_index])
                        grad_index += 1

                if module_grads:
                    block_size = self.optimal_block_sizes.get(name, 1)
                    adaptive_clip_value = self.compute_adaptive_clip_value(module_grads)
                    self.adaptive_clip_values[name] = adaptive_clip_value
                    private_module_grads = self.shuffle_and_clip_module_grads(
                        module_grads, block_size, adaptive_clip_value
                    )
                    private_grads.extend(private_module_grads)
                else:
                    private_grads.extend([None] * len(list(module.parameters())))

        # Record the adaptive clip values in the accountant
        self.accountant.record_clip_values(self.adaptive_clip_values)

        self.current_step += 1
        # Update the privacy spent without passing clip_values (already recorded)
        self.epsilon_spent, self.delta_spent = self.accountant.get_privacy_spent(self.current_step)

        return private_grads, self.epsilon_spent, self.delta_spent

    def compute_adaptive_clip_value(self, module_grads: List[torch.Tensor]) -> float:
        # Compute the mean norm of the module's gradients
        norms = [torch.norm(grad.detach(), p=2) for grad in module_grads]
        mean_norm = torch.mean(torch.stack(norms))
        adaptive_clip_value = self.clip_factor * mean_norm.item()
        return adaptive_clip_value

    def shuffle_and_clip_module_grads(
        self,
        module_grads: List[torch.Tensor],
        block_size: int,
        clip_value: float
    ) -> List[torch.Tensor]:
        return [
            self.shuffle_and_clip(grad, block_size, clip_value)
            for grad in module_grads
        ]

    def shuffle_and_clip(
        self,
        grad: torch.Tensor,
        block_size: int,
        clip_value: float
    ) -> torch.Tensor:
        flat_grad = grad.view(-1)
        num_elements = flat_grad.numel()
        num_blocks = math.ceil(num_elements / block_size)

        # Pad the gradient if necessary
        if num_elements % block_size != 0:
            padding = block_size - (num_elements % block_size)
            flat_grad = torch.cat(
                [flat_grad, torch.zeros(padding, device=flat_grad.device)]
            )

        # Reshape into blocks
        blocks = flat_grad.view(num_blocks, -1)

        # Vectorized clipping with adaptive clip value
        block_norms = torch.norm(blocks, dim=1, keepdim=True)
        scaling_factor = torch.clamp(clip_value / (block_norms + 1e-10), max=1.0)
        clipped_blocks = blocks * scaling_factor

        # Efficient in-place shuffling
        shuffled_indices = torch.randperm(num_blocks, device=blocks.device)
        shuffled_blocks = clipped_blocks[shuffled_indices]

        # Flatten and truncate to original size
        shuffled_grad = shuffled_blocks.view(-1)[:num_elements].view(grad.shape)

        return shuffled_grad

    def get_privacy_spent(self) -> Tuple[float, float]:
        return self.epsilon_spent, self.delta_spent

class DPShufflePrivacyAccountant:
    def __init__(
        self,
        model,
        target_epsilon: float,
        steps: int,
        base_clip_value: float,
        batch_size: int,
        dataset_size: int,
        eta: float = 1e-6
    ):
        self.model = model
        self.target_epsilon = target_epsilon
        self.steps = steps  # Total number of training steps
        self.base_clip_value = base_clip_value
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.eta = eta  # Small constant for numerical stability
        self.module_dimensions = self._get_module_dimensions()
        self.total_parameters = sum(self.module_dimensions.values())
        self.block_sizes = None  # To be optimized
        self.clip_history = []  # Stores clip values per iteration

    def _get_module_dimensions(self) -> Dict[str, int]:
        module_dimensions = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module) and list(module.parameters()):
                module_dimensions[name] = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
        return module_dimensions

    def compute_epsilon_i(
            self,
            d: int,
            beta: int,
            clip_value: float,
            batch_size: int,
            dataset_size: int
        ) -> float:
            if d == 0 or beta == 0:
                return 0.0

            q = batch_size / dataset_size
            c = clip_value
            m = d / beta

            epsilon_i = (c ** 2) * beta / (2 * m)
            epsilon_i_adjusted = q * epsilon_i

            return epsilon_i_adjusted

    def record_clip_values(self, clip_values: Dict[str, float]):
        """
        Records the clip values used at each iteration.

        Args:
            clip_values (Dict[str, float]): Dictionary of clip values per module.
        """
        self.clip_history.append(clip_values.copy())

    def compute_per_iteration_privacy(
        self,
        block_sizes: Dict[str, int],
        clip_values: Dict[str, float]
    ) -> float:
        """
        Computes the per-iteration privacy loss.

        Args:
            block_sizes (Dict[str, int]): Block sizes per module.
            clip_values (Dict[str, float]): Clip values per module.

        Returns:
            epsilon_iteration (float): Per-iteration privacy loss.
        """
        epsilons = [
            self.compute_epsilon_i(
                d=d_i,
                beta=block_sizes[module_name],
                clip_value=clip_values.get(module_name, self.base_clip_value),
                batch_size=self.batch_size,
                dataset_size=self.dataset_size
            )
            for module_name, d_i in self.module_dimensions.items()
        ]

        # Total per-iteration privacy loss
        epsilon_iteration = sum(epsilons)

        return epsilon_iteration

    def get_privacy_spent(self, step: int) -> Tuple[float, float]:
        epsilon_total = 0.0

        for t in range(step):
            clip_values = self.clip_history[t]
            epsilon_t = self.compute_per_iteration_privacy(self.block_sizes, clip_values)
            epsilon_total += epsilon_t

        delta_total = 0.0  # No delta in mutual information privacy

        return epsilon_total, delta_total

    def compute_total_privacy(
        self,
        block_sizes: Dict[str, int],
        clip_values: Dict[str, float]
    ) -> tuple:
        """
        Computes the total privacy loss over all steps.

        Args:
            block_sizes (Dict[str, int]): Block sizes per module.
            clip_values (Dict[str, float]): Clip values per module.

        Returns:
            epsilon_total (float): Total epsilon over all steps.
            delta_total (float): Total delta over all steps (set to 0).
        """
        # Compute per-iteration epsilon
        epsilon_iteration = self.compute_per_iteration_privacy(block_sizes, clip_values)

        # Use basic composition over all steps
        epsilon_total = self.steps * epsilon_iteration
        delta_total = 0.0  # No delta in mutual information privacy

        return epsilon_total, delta_total

    def optimize_parameters(self):
        """
        Optimizes block sizes to meet the target privacy budget.

        Returns:
            block_sizes (Dict[str, int]): Optimized block sizes per module.
        """
        self.block_sizes = self.find_optimal_block_sizes()
        return self.block_sizes

    def find_optimal_block_sizes(self) -> Dict[str, int]:
        def binary_search_global(target_epsilon_per_group):
            block_sizes = {}
            for module_name, d_i in self.module_dimensions.items():
                low, high = 1, d_i
                best_block_size = low
                while low <= high:
                    mid = (low + high) // 2
                    epsilon = self.compute_epsilon_i(
                        d_i,
                        mid,
                        self.base_clip_value,
                        self.batch_size,
                        self.dataset_size
                    )
                    if epsilon <= target_epsilon_per_group:
                        best_block_size = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                block_sizes[module_name] = best_block_size
            return block_sizes

        # Initialize search bounds for target_epsilon_per_group
        low, high = 0, self.target_epsilon / self.steps
        best_block_sizes = None
        best_epsilon_diff = float('inf')

        iterations = 0
        max_iterations = 1000

        while high - low > 1e-6 and iterations < max_iterations:
            mid = (low + high) / 2
            block_sizes = binary_search_global(mid)
            epsilon, _ = self.compute_total_privacy(
                block_sizes,
                {name: self.base_clip_value for name in self.module_dimensions}
            )

            if math.isinf(epsilon):
                high = mid
            else:
                epsilon_diff = abs(epsilon - self.target_epsilon)

                if epsilon_diff < best_epsilon_diff:
                    best_block_sizes = block_sizes.copy()
                    best_epsilon_diff = epsilon_diff

                if epsilon > self.target_epsilon:
                    high = mid
                else:
                    low = mid

            iterations += 1

        if best_block_sizes is None:
            raise ValueError("Unable to find valid block sizes. Try adjusting the privacy parameters.")

        return best_block_sizes

    def safe_exp(self, x):
        if x > 700:
            return float('inf')
        return math.exp(x)