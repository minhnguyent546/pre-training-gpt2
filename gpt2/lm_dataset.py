import glob
import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """A simple dataset for training models with language modeling objective.

    Dataset files should contain token ids, possibly divided into shards (.npy files),
    each shard then will be loaded lazily.
    """
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        seq_length: int,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        shard_files = glob.glob(os.path.join(dataset_dir, '*.npy'))
        if len(shard_files) == 0:
            raise ValueError(f'Could not find any .npy file in {dataset_dir}')
        if shuffle:
            random.shuffle(shard_files)
        else:
            shard_files = sorted(shard_files)
        self.shard_files = shard_files

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_replicas = num_replicas
        self.rank = rank

        self.token_per_batch = self.batch_size * self.seq_length
        self.shard_idx = -1
        self.ptr = self.rank * self.token_per_batch

        self._load_next_shard()

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Get the next batch."""
        input_ids = np.empty((self.token_per_batch,), dtype=np.uint16)
        labels = np.empty((self.token_per_batch,), dtype=np.uint16)
        num_tokens_to_fill = self.token_per_batch
        if self.ptr + (num_tokens_to_fill + 1) - 1 >= len(self.shard):
            # if we do not have enough tokens, take the remaining tokens and load the next shard
            num_remain_tokens = len(self.shard) - self.ptr - 1
            if num_remain_tokens > 0:
                input_ids[:num_remain_tokens] = self.shard[self.ptr:-1]
                labels[:num_remain_tokens] = self.shard[self.ptr + 1:]
                num_tokens_to_fill -= num_remain_tokens
            self.ptr = 0
            self._load_next_shard()

        # assume each shard contains no less than `num_tokens_to_fill + 1` tokens
        # TODO: handle this assumption
        assert num_tokens_to_fill + 1 <= len(self.shard)
        input_ids[-num_tokens_to_fill:] = self.shard[self.ptr:self.ptr + num_tokens_to_fill]
        labels[-num_tokens_to_fill:] = self.shard[self.ptr + 1:self.ptr + num_tokens_to_fill + 1]
        self.ptr = self.ptr + num_tokens_to_fill + self.token_per_batch * (self.num_replicas - 1)
        self._normalize_ptr()

        input_ids = torch.from_numpy(input_ids.astype(np.int32)).view(self.batch_size, self.seq_length)
        labels = torch.from_numpy(labels.astype(np.int64)).view(self.batch_size, self.seq_length)
        return input_ids, labels

    def _load_next_shard(self) -> None:
        self.shard_idx = (self.shard_idx + 1) % len(self.shard_files)
        self.shard = np.load(self.shard_files[self.shard_idx])

    def _normalize_ptr(self) -> None:
        """self.ptr may exceed the length of the shard, so we need to take care of that."""
        assert self.shard is not None
        # as we ignore the last token in each shard when filling input_ids,
        # so we need to subtract 1 from the length
        while self.ptr >= len(self.shard) - 1:
            self.ptr -= (len(self.shard) - 1)
            self._load_next_shard()
