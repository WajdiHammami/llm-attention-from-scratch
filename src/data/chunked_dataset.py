from src.data.tokenized_dataset import TokenizedDataset
from torch.utils.data import Dataset
import torch
import random
from tqdm import tqdm

class ChunkedDataset(Dataset):
    def __init__(self, dataset: TokenizedDataset, chunk_size: int = 512):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunked_tokens = []
        flattened = []
        print("Flattening dataset into chunks...")
        for text in tqdm(dataset.tokens):
            flattened.append(dataset.tokenizer.bos_id())  # Add BOS token before each text
            for token in text:
                flattened.append(token)
            flattened.append(dataset.tokenizer.eos_id())  # Add EOS token between texts

        self.all_tokens = torch.tensor(flattened, dtype=torch.long)
    
    def __len__(self):
        return len(self.all_tokens) // self.chunk_size
    
    def __getitem__(self, idx):
        # Add random offset within chunk for more variety
        random_offset = random.randint(0, max(0, len(self.all_tokens) - (idx + 1) * self.chunk_size - 1))
        idx += random_offset // self.chunk_size
        start_idx = idx * self.chunk_size
        end_idx = start_idx + self.chunk_size
        assert end_idx <= len(self.all_tokens), "Index out of range"
        input_ids = self.all_tokens[start_idx:end_idx]
        return {
            "input_ids": input_ids,
        }