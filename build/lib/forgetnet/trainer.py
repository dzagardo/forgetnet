# forgetnet/trainer.py
import torch
from trl import SFTTrainer
from typing import Dict, Any
from .dp.dp_shuffle import DPShuffleGenerator

class BloGSSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.target_epsilon = kwargs.pop('target_epsilon', 1.0)
        self.delta = kwargs.pop('delta', 1e-5)
        self.clip_value = kwargs.pop('clip_value', 1.0)
        self.neftune_noise_alpha = kwargs.pop('neftune_noise_alpha', None)
        super().__init__(*args, **kwargs)
        self.privacy_engine = self._create_privacy_engine()
        self.steps = 0

    def _create_privacy_engine(self):
        effective_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        return DPShuffleGenerator(
            model=self.model,
            target_epsilon=self.target_epsilon,
            delta=self.delta,
            steps=self.args.num_train_epochs * (len(self.train_dataset) // effective_batch_size),
            clip_value=self.clip_value,
            batch_size=effective_batch_size
        )

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        if (self.steps + 1) % self.args.gradient_accumulation_steps == 0:
            with torch.no_grad():
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                private_grads = self.privacy_engine.apply(grads)
                for param, private_grad in zip(model.parameters(), private_grads):
                    if param.grad is not None:
                        param.grad.copy_(private_grad)

        self.steps += 1
        return loss.detach()

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        
        privacy_spent = self.privacy_engine.get_privacy_spent()
        print(f"Final privacy budget spent: ε = {privacy_spent:.4f}")
        
        return result