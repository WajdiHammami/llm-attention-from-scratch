import torch
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training.train import Trainer
from src.config.training_config import TrainingConfig

# Simple mock model
class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=50, d_model=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, cache=None):
        h = self.embed(x)
        return self.linear(h)  # (B,T,vocab)


# Dataset that returns (input_ids)
class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, idx):
        x = torch.randint(0, 50, (8,))          # (T=8)
        return {
            "input_ids": x,
            "labels": x.clone()
        }


def test_gradient_accumulation_step_count():
    model = DummyModel()
    dataset = DummyDataset()

    config = TrainingConfig(
        batch_size=8,
        micro_batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
    )

    trainer = Trainer(model, dataset, config, device="cpu")

    # Count optimizer.step calls
    original_step = trainer.optimizer.step
    call_count = {"count": 0}

    def wrapped_step(*args, **kwargs):
        call_count["count"] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = wrapped_step

    trainer.train(eval_dataset=None)

    # Only one optimizer step should occur
    assert call_count["count"] == 1, \
        f"Expected 1 optimizer step, got {call_count['count']}"



def test_checkpoint_save_and_load(tmp_path):
    model = DummyModel()
    dataset = DummyDataset()

    config = TrainingConfig(
        batch_size=4,
        micro_batch_size=2,
        max_steps=1,
        output_dir=str(tmp_path),
        warmup_steps=0
    )

    trainer = Trainer(model, dataset, config, device="cpu")

    trainer.train(eval_dataset=None)

    # Check file exists
    saved = list(tmp_path.iterdir())
    assert len(saved) > 0, "Checkpoint was not saved"

    # Try loading
    ckpt_path = saved[0]
    step = trainer._load_checkpoint(str(ckpt_path))

    assert step == 0, "Checkpoint did not restore correct step number"


def test_evaluate_loop_runs():
    model = DummyModel()
    dataset = DummyDataset()

    trainer = Trainer(model, dataset, TrainingConfig(), device="cpu")

    loss = trainer.evaluate(dataset)

    assert isinstance(loss, float)
    assert loss > 0


def test_batch_device_transfer():
    model = DummyModel()
    dataset = DummyDataset()

    trainer = Trainer(model, dataset, TrainingConfig(), device="cpu")

    batch = next(iter(trainer.dataloader))
    batch = {k: v.to(trainer.device) for k, v in batch.items()}

    for v in batch.values():
        assert v.device.type == "cpu"



def test_shift_logits_and_labels():
    model = DummyModel(vocab_size=50)
    dataset = DummyDataset()

    trainer = Trainer(model, dataset, TrainingConfig(), device="cpu")

    batch = dataset[0]
    batch = {k: v.unsqueeze(0) for k, v in batch.items()}  # B=1

    loss = trainer._compute_loss(batch)

    # ensure shapes match expectations
    B, T = batch["input_ids"].shape
    logits = model(batch["input_ids"], cache=None)

    assert logits.shape == (1, T, 50)
    assert loss >= 0


