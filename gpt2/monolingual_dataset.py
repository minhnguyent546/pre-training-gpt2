from typing import Any

from tokenizers import Tokenizer

import numpy as np

from torch.utils.data import Dataset


class MonolingualDataset(Dataset):
    def __init__(
        self,
        data: np.memmap[Any, Any],
        seq_length: int,
        tokenizer: Tokenizer,
        random_item: bool = True,
    ):
        self.data = data
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.random_item = random_item
        self.data_length = len(self.data)

    def __len__(self) -> int:
        return self.data_length - self.seq_length

    def __getitem__(self, idx: int):
        if self.random_item:
            idx = np.random.randint(0, len(self))
        return {
            'input_ids': self.data[idx:idx+self.seq_length].copy(),
            'labels': self.data[idx+1:idx+1+self.seq_length].copy(),
        }
