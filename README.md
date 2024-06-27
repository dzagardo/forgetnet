# forgetnet

A package for applying differential privacy to model weights.

## Installation

```bash
pip install forgetnet
```
```
from forgetnet.dp_weights import calculate_noise_scale_poly, apply_noise_to_all_weights

```

# Example usage
```
model = ...  # Your PyTorch model
epsilon = 1.0
delta = 1e-5
clipping_norm = 1.0
dataset_size = 10000
batch_size = 32
num_epochs = 10
learning_rate = 0.001

apply_noise_to_all_weights(
    model, 
    calculate_noise_scale_poly, 
    epsilon, 
    delta, 
    clipping_norm, 
    dataset_size, 
    batch_size, 
    num_epochs, 
    learning_rate
)

```