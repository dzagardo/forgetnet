# forgetnet/dp/dp_shuffle.py

import math
from typing import List, Tuple
import torch
from ..core import PrivacyMechanism

class DPShufflePrivacyAccountant:
    def __init__(self, model, target_epsilon, delta, steps, clip_value, batch_size):
        self.model = model
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps = steps
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.parameter_dimensions = [p.numel() for p in model.parameters() if p.requires_grad]
        self.total_parameters = sum(self.parameter_dimensions)
        self.block_sizes = None

    def compute_epsilon_i(self, d_i: int, block_size: int) -> float:
        C = self.clip_value
        B = self.batch_size

        epsilon_1 = 2 * math.log(1 + d_i * (math.exp(2 * C / (math.sqrt(d_i))) - 1))
        epsilon_2 = 2 * math.log(1 + (block_size / d_i) * (math.exp(2 * C * math.sqrt(block_size / d_i)) - 1))
        
        return min(epsilon_1, epsilon_2)

    def compute_total_privacy(self, block_sizes: List[int]) -> float:
        epsilons = [self.compute_epsilon_i(d_i, block_size) 
                    for d_i, block_size in zip(self.parameter_dimensions, block_sizes)]
        
        epsilon_total_per_step = sum(epsilons)
        
        if epsilon_total_per_step > 700:
            return float('inf')
        
        epsilon_total = math.sqrt(2 * self.steps * math.log(1/self.delta)) * epsilon_total_per_step + \
                        self.steps * epsilon_total_per_step * (math.exp(epsilon_total_per_step) - 1)

        return epsilon_total

    def find_optimal_block_sizes(self) -> List[int]:
        def binary_search_global(target_epsilon_per_group):
            block_sizes = []
            for d_i in self.parameter_dimensions:
                low, high = 1, d_i - 1
                best_block_size = low
                while low <= high:
                    mid = (low + high) // 2
                    epsilon = self.compute_epsilon_i(d_i, mid)
                    if epsilon <= target_epsilon_per_group:
                        best_block_size = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                block_sizes.append(best_block_size)
            return block_sizes

        low, high = 0, self.target_epsilon / self.steps
        best_block_sizes = None
        best_epsilon_diff = float('inf')

        while high - low > 1e-6:
            mid = (low + high) / 2
            block_sizes = binary_search_global(mid)
            epsilon = self.compute_total_privacy(block_sizes)
            epsilon_diff = abs(epsilon - self.target_epsilon)

            if epsilon_diff < best_epsilon_diff:
                best_block_sizes = block_sizes
                best_epsilon_diff = epsilon_diff

            if epsilon > self.target_epsilon:
                high = mid
            else:
                low = mid

        print(f"Optimized epsilon: {self.compute_total_privacy(best_block_sizes):.4f}, target_epsilon: {self.target_epsilon}")
        return best_block_sizes

    def optimize_parameters(self):
        self.block_sizes = self.find_optimal_block_sizes()
        return self.block_sizes

class DPShuffleGenerator(PrivacyMechanism):
    def __init__(self, model: torch.nn.Module, target_epsilon: float, delta: float, steps: int, clip_value: float, batch_size: float):
        self.model = model
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps = steps
        self.clip_value = clip_value
        self.accountant = DPShufflePrivacyAccountant(model, target_epsilon, delta, steps, clip_value, batch_size)
        self.optimal_block_sizes = self.accountant.optimize_parameters()
        print(f"Optimal block sizes: {self.optimal_block_sizes}")
        self.epsilon_spent = 0

    def apply(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        private_grads, _, _ = self.generate(gradients)
        return private_grads

    def generate(self, gradients: List[torch.Tensor]) -> Tuple[List[torch.Tensor], float, float]:
        private_grads = []
        for grad, block_size in zip(gradients, self.optimal_block_sizes):
            clipped_grad = self.clip_gradient(grad)
            private_grad = self.shuffle(clipped_grad, block_size)
            private_grads.append(private_grad)

        self.epsilon_spent = self.accountant.compute_total_privacy(self.optimal_block_sizes)

        return private_grads, self.epsilon_spent, self.delta

    def shuffle(self, grad: torch.Tensor, block_size: int) -> torch.Tensor:
        flat_grad = grad.view(-1)
        num_elements = flat_grad.numel()
        num_blocks = math.ceil(num_elements / block_size)

        # Pad the gradient if necessary
        if num_elements % block_size != 0:
            padding = block_size - (num_elements % block_size)
            flat_grad = torch.cat([flat_grad, torch.zeros(padding, device=flat_grad.device)])

        # Reshape into blocks
        blocks = flat_grad.view(num_blocks, -1)

        # Shuffle the blocks
        shuffled_indices = torch.randperm(num_blocks, device=blocks.device)
        shuffled_blocks = blocks[shuffled_indices]

        # Flatten and truncate to original size
        shuffled_grad = shuffled_blocks.view(-1)[:num_elements]

        return shuffled_grad.view(grad.shape)

    def clip_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.norm(grad)
        factor = min(1, self.clip_value / grad_norm)
        return grad * factor

    def get_privacy_spent(self) -> float:
        return self.epsilon_spent