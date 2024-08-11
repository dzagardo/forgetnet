# forgetnet/dp/dp_shuffle.py

import math
from typing import List, Tuple
import torch
from ..core import PrivacyMechanism
from transformers import modeling_utils
import torch.nn as nn

class DPShufflePrivacyAccountant:
    def __init__(self, model, target_epsilon, delta, steps, clip_value, batch_size):
        self.model = model
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps = steps
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.module_dimensions = [sum(p.numel() for p in module.parameters() if p.requires_grad)
                                  for module in model.modules() if self._is_supported_module(module)]
        self.total_parameters = sum(self.module_dimensions)
        self.block_sizes = None

    def _is_supported_module(self, module):
        return isinstance(module, (
            # Common layers
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Embedding,
            
            # Normalization layers
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            
            # Recurrent layers
            nn.LSTM,
            nn.GRU,
            nn.RNN,
            
            # Attention mechanisms
            nn.MultiheadAttention,
            
            # Activation functions with parameters
            nn.PReLU,
            
            # Transformer-specific modules
            modeling_utils.Conv1D,
        ))

    def compute_epsilon_i(self, d_i: int, block_size: int) -> float:
        C = self.clip_value

        if d_i == 0:
            return 0.0

        epsilon_1 = 2 * math.log(1 + d_i * (math.exp(2 * C / (math.sqrt(d_i))) - 1))
        epsilon_2 = 2 * math.log(1 + (block_size / d_i) * (math.exp(2 * C * math.sqrt(block_size / d_i)) - 1))
        
        return min(epsilon_1, epsilon_2)

    def compute_total_privacy(self, block_sizes: List[int]) -> float:
        epsilons = [self.compute_epsilon_i(d_i, block_size) 
                    for d_i, block_size in zip(self.module_dimensions, block_sizes)]
        
        epsilon_total_per_step = sum(epsilons)
        
        if epsilon_total_per_step > 700:
            return float('inf')
        
        epsilon_total = math.sqrt(2 * self.steps * math.log(1/self.delta)) * epsilon_total_per_step + \
                        self.steps * epsilon_total_per_step * (math.exp(epsilon_total_per_step) - 1)

        return epsilon_total

    def find_optimal_block_sizes(self) -> List[int]:
        def binary_search_global(target_epsilon_per_group):
            block_sizes = []
            for d_i in self.module_dimensions:
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
        self.module_to_block_size = {module: block_size for module, block_size in zip(self._get_supported_modules(), self.optimal_block_sizes)}
        print(f"Optimal block sizes: {self.optimal_block_sizes}")
        self.epsilon_spent = 0

    def _get_supported_modules(self):
        return [module for module in self.model.modules() if self._is_supported_module(module)]

    def _is_supported_module(self, module):
        return isinstance(module, (
            # Common layers
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Embedding,
            
            # Normalization layers
            nn.LayerNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            
            # Recurrent layers
            nn.LSTM,
            nn.GRU,
            nn.RNN,
            
            # Attention mechanisms
            nn.MultiheadAttention,
            
            # Activation functions with parameters
            nn.PReLU,
            
            # Transformer-specific modules
            modeling_utils.Conv1D,
        ))

    def apply(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        private_grads, _, _ = self.generate(gradients)
        return private_grads

    def generate(self, gradients: List[torch.Tensor], modules: List[nn.Module]) -> Tuple[List[torch.Tensor], float, float]:
        private_grads = []
        for grad, module in zip(gradients, modules):
            block_size = self.module_to_block_size[module]
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