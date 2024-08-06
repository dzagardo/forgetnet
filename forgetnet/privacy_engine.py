# forgetnet/privacy_engine.py

import torch
import torch.optim as optim
import torch.nn as nn
from .dp import DPShuffleGenerator
import logging

logger = logging.getLogger(__name__)

class BloGSPrivacyEngine:
    def __init__(self, optimizer: optim.Optimizer, model: nn.Module, 
                 target_epsilon: float, delta: float, clip_value: float, 
                 steps: int, batch_size: int):
        self.optimizer = optimizer
        self.model = model
        self.dp_generator = DPShuffleGenerator(
            model=model,
            target_epsilon=target_epsilon,
            delta=delta,
            clip_value=clip_value,
            steps=steps,
            batch_size=batch_size
        )
        
    def step(self):
        try:
            with torch.no_grad():
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                if not grads:
                    logger.warning("No gradients found. Skipping privacy step.")
                    return 0, 0

                private_grads, epsilon_spent, delta = self.dp_generator.generate(grads)
                
                for param, private_grad in zip(self.model.parameters(), private_grads):
                    if param.grad is not None:
                        if isinstance(private_grad, torch.Tensor):
                            if private_grad.shape != param.grad.shape:
                                logger.warning(f"Shape mismatch: param.grad.shape = {param.grad.shape}, private_grad.shape = {private_grad.shape}")
                                private_grad = private_grad.reshape(param.grad.shape)
                            param.grad.data = private_grad
                        else:
                            logger.error(f"Unexpected private_grad type: {type(private_grad)}")
                            raise TypeError(f"Expected torch.Tensor, got {type(private_grad)}")

            self.optimizer.step()
            return epsilon_spent, delta
        except Exception as e:
            logger.error(f"Error in privacy step: {str(e)}")
            raise

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_privacy_spent(self):
        return self.dp_generator.get_privacy_spent()