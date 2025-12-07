from datasets import load_dataset
import json
import argparse


    
def collecting_function(ds):
    collected = []
    MAX_SIZE = 100 * 1024 * 1024  # 100 MB
    current_size = 0

    for sample in ds:
        collected.append(sample["text"])
        current_size += len(sample["text"].encode("utf-8"))
        if current_size >= MAX_SIZE:
            break

if __name__ == "__main__":

    print("Downloading dataset...")

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    
    with open("src/data/raw/wikitext.txt", "w") as f:
        for item in ds["train"]["text"]:
            f.write(item)
    print("Saved to src/data/raw/wikitext.txt")
    
