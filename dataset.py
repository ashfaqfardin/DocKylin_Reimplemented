import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class FUNSDDataset(Dataset):
    """Simple dataset for pre-training. Returns image and full text."""
    def __init__(self, split='train'):
        self.dataset = load_dataset("nielsr/funsd", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"image": item['image'], "text": " ".join(item['words'])}


class FUNSDInstructionDataset(Dataset):
    """Dataset for instruction-tuning. Returns image, instruction, and answer."""
    def __init__(self, split='train'):
        self.dataset = load_dataset("nielsr/funsd", split=split)
        self.instructions = [
            "Transcribe the document.",
            "What is the text content of the attached image?",
            "Read the document and provide the full text.",
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "image": item['image'],
            "instruction": np.random.choice(self.instructions),
            "answer": " ".join(item['words'])
        }