import torch


class DataCollator:
    def __init__(self, pad_token_id=3, block_size=512):
        self.pad_token_id = pad_token_id
        self.block_size = block_size

    def __call__(self, samples):
        seq = []
        for sample in samples:
            if sample["input_ids"].shape[0] < self.block_size:
                pad_length = self.block_size - sample["input_ids"].shape[0]
                padding = torch.full((pad_length,), self.pad_token_id, dtype=torch.long) 
                sample["input_ids"] = torch.cat([sample["input_ids"], padding], dim=0)
            seq.append(sample["input_ids"])
        input_ids = torch.stack(seq, dim=0)  # (B, T)

        return {
            "input_ids": input_ids,
        }