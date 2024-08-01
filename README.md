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

## 🚀 Quick Start

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