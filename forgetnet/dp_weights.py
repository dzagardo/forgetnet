# my_dp_model_package/dp_weights.py

import numpy as np
import torch

def calculate_noise_scale_poly(epsilon, delta, clipping_norm, batch_size, learning_rate, num_epochs, dataset_size, params=None, **kwargs):
    # Default parameters if none are provided
    if params is None:
        params = [0.033599, 0.589936, 0.099315, 0.007782, 4.983281, 8.659760]
    c, k1, k2, k3, k4, k5 = params
    c = 1
    k1 = 1
    k2 = 1
    k3 = 0.009760
    k4 = 0.078008
    # k5 = 0.002486
    delta = 1 / dataset_size ** 2  # Fixed delta value
    return c * np.sqrt(2 * np.log(1.25 / delta)) * \
           ((k1 * num_epochs * learning_rate * clipping_norm) / ((epsilon ** k2))) / (dataset_size * batch_size) + (k3 / epsilon ** k4)

def apply_noise_to_all_weights(model, noise_scale_fn, epsilon, delta, clipping_norm, dataset_size, batch_size, num_epochs, learning_rate=5e-5, **kwargs):

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
