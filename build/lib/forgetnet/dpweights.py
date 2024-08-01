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
    def apply_noise(model_or_params, noise_scale_fn, epsilon, delta, clipping_norm, dataset_size, batch_size, num_epochs, learning_rate=5e-5, lora_config=None, **kwargs):
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

        if isinstance(model_or_params, (list, torch.nn.ParameterList)):
            params_to_noise = model_or_params
        else:
            if lora_config:
                # If LoRA config is provided, only apply noise to LoRA parameters
                params_to_noise = [p for n, p in model_or_params.named_parameters() if 'lora' in n]
            else:
                params_to_noise = model_or_params.parameters()

        for param in params_to_noise:
            original_dtype = param.dtype
            original_device = param.device

            param_float = param.data.float()
            noise = torch.normal(0, noise_scale, param_float.shape).to(original_device)
            noisy_weights = param_float + noise
            param.data.copy_(noisy_weights.to(original_dtype))