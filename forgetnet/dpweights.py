# forgetnet/dp_weights.py

import numpy as np
import torch

class DifferentialPrivacyWeights:
    @staticmethod
    def calculate_noise_scale(epsilon, delta, clipping_norm, batch_size, learning_rate, num_epochs, dataset_size, params=None):
        """
        Calculate the noise scale for differential privacy.
        
        Parameters:
        - epsilon: Privacy budget
        - delta: Probability of not preserving privacy
        - clipping_norm: The norm to which gradients are clipped
        - batch_size: Size of the batches used during training
        - learning_rate: Learning rate used during training
        - num_epochs: Number of epochs used during training
        - dataset_size: Size of the dataset used for training
        
        Returns:
        - noise_scale: Calculated noise scale
        """
        
        c, k1, k2, k3, k4 = 1, 1, 1, 0.009760, 0.078008  # Fixed values

        delta = 1 / dataset_size ** 2  # Fixed delta value
        noise_scale = (c * np.sqrt(2 * np.log(1.25 / delta)) * 
                      ((k1 * num_epochs * learning_rate * clipping_norm) / (epsilon ** k2)) /
                      (dataset_size * batch_size) + (k3 / epsilon ** k4))
        return noise_scale

    @staticmethod
    def apply_noise(model, noise_scale_fn, epsilon, delta, clipping_norm, dataset_size, batch_size, num_epochs, learning_rate=5e-5, **kwargs):
        """
        Apply noise to all weights of the model for differential privacy.
        
        Parameters:
        - model: The PyTorch model to apply noise to
        - noise_scale_fn: Function to calculate the noise scale
        - epsilon: Privacy budget
        - delta: Probability of not preserving privacy
        - clipping_norm: The norm to which gradients are clipped
        - dataset_size: Size of the dataset used for training
        - batch_size: Size of the batches used during training
        - num_epochs: Number of epochs used during training
        - learning_rate: Learning rate used during training
        """
        noise_scale = noise_scale_fn(
            epsilon=epsilon, 
            delta=delta, 
            clipping_norm=clipping_norm, 
            batch_size=batch_size, 
            num_epochs=num_epochs, 
            learning_rate=learning_rate, 
            dataset_size=dataset_size,
            **kwargs
        )

        for name, param in model.named_parameters():
            noise = np.random.normal(0, noise_scale, param.data.cpu().numpy().shape)
            noisy_weights = param.data.cpu().numpy() + noise
            param.data.copy_(torch.tensor(noisy_weights).to(param.device))
