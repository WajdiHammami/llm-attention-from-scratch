import os
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as F
import json
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        training_config,
        collator,
        device='cuda'
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.config = training_config
        self.device = device

        # intialize other components like optimizer, loss function, etc.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )


        self.scaler = torch.amp.GradScaler(enabled=(self.config.precision in ["fp16", "bf16"]))
        # Set up Dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collator
        )

        # Other initializations
        torch.manual_seed(self.config.seed)
        self.start_step = 0

        if self.config.resume_from:
            self.start_step = self._load_checkpoint(self.config.resume_from)
        
        # val loader can be set up similarly if needed
        self.val_loader = None
        self.collator = collator
    def _compute_loss(self, batch):
        input_ids = batch["input_ids"]      # (B,T)
        logits = self.model(input_ids, cache=None)
        
        targets = input_ids[:, 1:].clone()
        logits = logits[:, :-1, :]
        # Mask
        targets[input_ids[:, :-1] == self.config.pad_token_id] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), label_smoothing=0.1, ignore_index=-100)
        return loss

    def _train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        accumulation_steps = self.config.batch_size // self.config.micro_batch_size
        total_loss = 0.0
        for micro_step in range(accumulation_steps):
            micro_batch = {k: v[micro_step*self.config.micro_batch_size:(micro_step+1)*self.config.micro_batch_size] for k, v in batch.items()}
            with torch.autocast(device_type="cuda", enabled=(self.config.precision in ["fp16", "bf16"]), dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16):
                loss = self._compute_loss(micro_batch) / accumulation_steps
            self.scaler.scale(loss).backward()
            total_loss += loss.item() / self.config.batch_size
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return total_loss 

    def train(self, eval_dataset=None):
        self.model.train()
        data_iter = iter(self.dataloader)
        for step in tqdm(range(0, self.config.max_steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self._train_step(batch)

            if step % self.config.log_interval == 0:
                # how much of an epoch we've completed
                epoch_progress = (step * self.config.batch_size) / len(self.dataset)
                self._log(f"Step {step}: Training loss: {loss}, Epoch progress: {epoch_progress:.2f}")
            
            if eval_dataset is not None and step % self.config.eval_interval == 0:
                eval_loss = self.evaluate(eval_dataset)
                self._log(f"Step {step}: Evaluation loss: {eval_loss}")
            
            if step % self.config.save_interval == 0:
                self._save_checkpoint(self.config.output_dir, step)

    def evaluate(self, val_dataset):
        self.model.eval()
        average_loss = 0.0
        if self.val_loader is None:
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collator
            )
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                loss = self._compute_loss(batch)
            average_loss += loss.item()
        average_loss /= len(self.val_loader)
        
        return average_loss / self.config.batch_size

    def _save_checkpoint(self, path, step:int):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': step,
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
        }, f"{path}/checkpoint_step_{step}.pt")


    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        # Reload config from dict if needed
        self.start_step = checkpoint['step']
        return checkpoint['step']

    def _log(self, message: str):
        tqdm.write(message)
        os.makedirs(self.config.log_dir, exist_ok=True)

        with open(os.path.join(self.config.log_dir, "training_log.jsonl"), "a") as f:
            json.dump({"step": self.start_step, "message": message}, f)
            f.write("\n")