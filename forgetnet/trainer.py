# forgetnet/trainer.py
import torch
from trl import SFTTrainer
from typing import Optional, Dict, Any
from .dp.dp_shuffle import DPShuffleGenerator

class DPBloGSTrainer(SFTTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        args: Any,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        optimizers: Optional[Any] = (None, None),
        callbacks: Optional[Any] = None,
        target_epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_value: float = 1.0,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks=callbacks,
        )
        
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.clip_value = clip_value
        self.privacy_engine = self._create_privacy_engine()
        self.steps = 0

    def _create_privacy_engine(self):
        return DPShuffleGenerator(
            model=self.model,
            target_epsilon=self.target_epsilon,
            delta=self.delta,
            steps=self.args.num_train_epochs * (len(self.train_dataset) // self.args.train_batch_size),
            clip_value=self.clip_value,
            batch_size=self.args.train_batch_size
        )

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

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