# 🛡️ ForgetNet: Differentially Private Block-wise Gradient Shuffle for Deep Learning 🧠

[![PyPI version](https://badge.fury.io/py/forgetnet.svg)](https://badge.fury.io/py/forgetnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ForgetNet introduces a novel privacy-preserving technique for deep learning: Differentially Private Block-wise Gradient Shuffle (DP-BloGS). 🔒🔀

## 🌟 Features

- 🚀 Fast training times, close to non-private training
- 🎯 Competitive privacy guarantees compared to DP-SGD
- 📊 Better privacy-utility trade-off in many scenarios
- 🏋️ Scalable to large models (tested up to 1.1 billion parameters)
- 🧮 Parameter-wise privacy budget allocation

## 📦 Installation

```
pip install forgetnet
```

## 🚀 Quick Start BloGSSFTTrainer

```python
from forgetnet import BloGSSFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize the DP-BloGS trainer
trainer = BloGSSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    target_epsilon=1.0,
    delta=1e-5,
    clip_value=1.0
)

# Train your model with privacy guarantees
trainer.train()
```

## 🚀 Quick Start with Privacy Engine and ResNet

Here's how to use the BloGS Privacy Engine with a ResNet model for MNIST classification:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from forgetnet import BloGSPrivacyEngine

# Modify ResNet18 for MNIST (1 channel input instead of 3)
def mnist_resnet18():
    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for MNIST
    return model

# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 10
target_epsilon = 1.0
delta = 1e-5
clip_value = 1.0

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Calculate total_iterations based on dataset size
total_iterations = (len(train_dataset) // batch_size) * epochs

# Initialize model
model = mnist_resnet18().to(device)

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Wrap the optimizer with the PrivacyEngine
privacy_engine = BloGSPrivacyEngine(
    optimizer=optimizer,
    model=model,
    target_epsilon=target_epsilon,
    delta=delta,
    clip_value=clip_value,
    steps=total_iterations,
    batch_size=batch_size
)

# Training loop
model.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        privacy_engine.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()

        epsilon_spent, delta = privacy_engine.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Epsilon: {epsilon_spent:.4f}')

# Get total privacy spent after training
total_epsilon_spent = privacy_engine.get_privacy_spent()
print(f"Total privacy spent: ε = {total_epsilon_spent:.4f}")
```

This example demonstrates:

1. Setting up a ResNet18 model modified for MNIST
2. Loading the MNIST dataset
3. Initializing the PrivacyEngine with the model and optimizer
4. Training the model with privacy-preserving gradient updates
5. Monitoring privacy budget expenditure during training

Adjust hyperparameters as needed for your specific use case.

## 📚 How It Works

DP-BloGS introduces a probabilistic approach to gradient noise through block-wise shuffling:

1. 📊 Divide gradients into blocks
2. 🔀 Shuffle blocks randomly
3. 📏 Apply parameter-specific block sizes
4. ✂️ Use batch layer clipping
5. 🧮 Accumulate gradients

This combination allows for fast training while maintaining strong privacy guarantees!

## 📈 Performance

DP-BloGS has been tested on various model architectures, including:

- GPT-2 (124M)
- BERT (110M)
- OPT (350M)
- BLOOM (560M)
- TinyLlama (1.1B)

Results show competitive or better performance compared to DP-SGD in terms of:

- 🏃‍♂️ Training speed
- 🎭 Privacy guarantees
- 📊 Model utility

## 📄 Citation

If you use ForgetNet in your research, please cite my paper:

```bibtex
@article{zagardo2024dpblogs,
  title={Differentially Private Block-wise Gradient Shuffle for Deep Learning},
  author={Zagardo, David},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see my [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

We thank the open-source community and the authors of the papers cited in our work for their valuable contributions to the field of privacy-preserving machine learning.

---

Built with 🧠 by David Zagardo