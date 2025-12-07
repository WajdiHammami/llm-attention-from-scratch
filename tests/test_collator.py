import torch
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.tokenized_dataset import TokenizedDataset
from src.data.chunked_dataset import ChunkedDataset
from src.data.collator import DataCollator

def test_chunked_dataset_basic():
    # Fake tokenized dataset
    class DummyTok:
        tokens = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]

    dataset = DummyTok()
    chunk_size = 4
    chunked = ChunkedDataset(dataset, chunk_size)

    # Flattened tokens = [1,2,3,4,5,6,7,8,9]
    # Expected chunks:
    #   [1,2,3,4]
    #   [5,6,7,8]
    # Leftover = [9] → ignored

    assert len(chunked) == 2

    first = chunked[0]["input_ids"]
    second = chunked[1]["input_ids"]

    assert torch.equal(first, torch.tensor([1,2,3,4]))
    assert torch.equal(second, torch.tensor([5,6,7,8]))


def test_chunked_dataset_exact_multiple():
    class DummyTok:
        tokens = [
            [10, 20],
            [30, 40]
        ]

    dataset = DummyTok()
    chunked = ChunkedDataset(dataset, chunk_size=2)

    assert len(chunked) == 2
    assert torch.equal(chunked[0]["input_ids"], torch.tensor([10,20]))
    assert torch.equal(chunked[1]["input_ids"], torch.tensor([30,40]))



def test_collator_padding_and_stacking():
    collator = DataCollator(pad_token_id=3, block_size=8)

    samples = [
        {"input_ids": torch.tensor([1, 2, 3])},
        {"input_ids": torch.tensor([4, 5, 6, 7, 8])},
    ]

    batch = collator(samples)
    input_ids = batch["input_ids"]

    # Shape must be (B, block_size)
    assert input_ids.shape == (2, 8)

    # First sample padded with PAD token = 3
    assert torch.equal(
        input_ids[0],
        torch.tensor([1,2,3,3,3,3,3,3])
    )

    # Second sample padded (block_size = 8)
    assert torch.equal(
        input_ids[1],
        torch.tensor([4,5,6,7,8,3,3,3])
    )