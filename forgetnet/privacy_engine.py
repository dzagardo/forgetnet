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
                 steps: int, batch_size: int, dataset_size: int = None):
        self.optimizer = optimizer
        self.model = model
        self.generator = DPShuffleGenerator(
            model=model,
            target_epsilon=target_epsilon,
            delta=delta,
            clip_value=clip_value,
            steps=steps,
            batch_size=batch_size,
            dataset_size=dataset_size
        )
        self.steps = 0

    def step(self):
        with torch.no_grad():
            # Collect gradients from all model parameters
            grads = [param.grad for param in self.model.parameters() if param.grad is not None]

            if not grads:
                logger.warning("No gradients to process in PrivacyEngine.step()")
                self.optimizer.step()
                return None, None

            # Generate private gradients
            private_grads, epsilon_spent, delta_spent = self.generator.generate(grads)

            # Assign private gradients back to model parameters
            grad_iter = iter(private_grads)
            for param in self.model.parameters():
                if param.grad is not None:
                    private_grad = next(grad_iter, None)
                    if private_grad is not None:
                        if private_grad.shape != param.grad.shape:
                            logger.warning(
                                f"Shape mismatch for parameter '{param.name if hasattr(param, 'name') else 'unknown'}': "
                                f"param.grad.shape = {param.grad.shape}, private_grad.shape = {private_grad.shape}. "
                                f"Reshaping private_grad."
                            )
                            private_grad = private_grad.view_as(param.grad)
                        param.grad.copy_(private_grad)
                    else:
                        logger.error(f"Private gradient for parameter '{param.name if hasattr(param, 'name') else 'unknown'}' is None.")
            
        # Perform the optimization step
        self.optimizer.step()

        # Increment step counter
        self.steps += 1

        return epsilon_spent, delta_spent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_privacy_spent(self):
        return self.generator.get_privacy_spent()
