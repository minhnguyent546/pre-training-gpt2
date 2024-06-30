import datasets
from tokenizers import Tokenizer

import numpy as np

from torch.utils.data import Dataset


class MonolingualDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        seq_length: int,
        tokenizer: Tokenizer,
        random_item: bool = True,
    ):
        self.dataset = dataset
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.random_item = random_item

        self.ids = np.concatenate(self.dataset['ids'])

    def __len__(self) -> int:
        return len(self.dataset) - self.seq_length

    def __getitem__(self, idx: int):
        if self.random_item:
            idx = np.random.randint(0, len(self))
        return {
            'input_ids': self.ids[idx:idx+self.seq_length],
            'labels': self.ids[idx+1:idx+1+self.seq_length]
        }
