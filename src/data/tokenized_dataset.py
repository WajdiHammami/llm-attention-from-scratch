from src.data.tokenizer import get_tokenizer, encode, decode, vocab_size
from torch.utils.data import Dataset
import torch
from tqdm import tqdm




class TokenizedDataset(Dataset):
    def __init__(self, data_path: str):
        self.tokenizer = get_tokenizer()

        with open(data_path, "r") as f:
            lines = f.readlines()
            print(f"Loaded {len(lines)} lines from {data_path}")
            print("Tokenizing dataset...")
            self.tokens = [encode(line.strip(), self.tokenizer) for line in tqdm(lines)]
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        input_ids = self.tokens[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
        }
