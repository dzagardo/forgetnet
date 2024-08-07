# forgetnet/privacy_engine.py

import torch
import torch.optim as optim
import torch.nn as nn
from .dp import DPShuffleGenerator
import logging
from transformers import modeling_utils

logger = logging.getLogger(__name__)

class BloGSPrivacyEngine:
    def __init__(self, optimizer: optim.Optimizer, model: nn.Module, 
                 target_epsilon: float, delta: float, clip_value: float, 
                 steps: int, batch_size: int):
        self.optimizer = optimizer
        self.model = model
        self.generator = DPShuffleGenerator(
            model=model,
            target_epsilon=target_epsilon,
            delta=delta,
            clip_value=clip_value,
            steps=steps,
            batch_size=batch_size
        )
        self.module_to_name_map = self._create_module_to_name_map()

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

    def _create_module_to_name_map(self):
        module_to_name_map = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (
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
            )):
                module_to_name_map[module] = name
        return module_to_name_map

    def step(self):
        with torch.no_grad():
            grads_modules_names = []
            for module in self.model.modules():
                if self._is_supported_module(module):
                    for param in module.parameters():
                        if param.grad is not None:
                            grads_modules_names.append((param.grad, module, self.module_to_name_map[module]))
            
            grads, modules, layer_names = zip(*grads_modules_names)
          
            private_grads, epsilon_spent, delta = self.generator.generate(list(grads), list(modules))
            
            index = 0
            for module in self.model.modules():
                if self._is_supported_module(module):
                    for param in module.parameters():
                        if param.grad is not None:
                            private_grad = private_grads[index]
                            if isinstance(private_grad, torch.Tensor):
                                if private_grad.shape != param.grad.shape:
                                    print(f"Shape mismatch: param.grad.shape = {param.grad.shape}, private_grad.shape = {private_grad.shape}")
                                    private_grad = private_grad.reshape(param.grad.shape)
                                param.grad.data = private_grad
                            else:
                                print(f"Unexpected private_grad type: {type(private_grad)}")
                            index += 1

        self.optimizer.step()
        return epsilon_spent, delta

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_privacy_spent(self):
        return self.dp_generator.get_privacy_spent()