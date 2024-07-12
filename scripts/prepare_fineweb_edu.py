#!/usr/bin/env python

"""
Download and tokenize the fineweb-edu dataset (10BT subset) (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
for pre-training GPT2.
"""

import argparse
import os
import random
from typing import Any

import datasets
import numpy as np
import tiktoken
from tqdm.autonotebook import tqdm


def prepare_fineweb_edu(args: argparse.Namespace) -> None:
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # tokenizer
    tokenizer = tiktoken.encoding_for_model(args.tokenizer)
    eot_id = tokenizer.eot_token
    dump_id = -1

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_workers = args.num_workers

    ds = datasets.load_dataset(
        'HuggingFaceFW/fineweb-edu',
        'sample-10BT',
        num_proc=num_workers,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    def tokenize_examples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tokens_list = tokenizer.encode_ordinary_batch(examples['text'], num_threads=num_workers)
        tokens_list = [tokens + [eot_id] for tokens in tokens_list]
        lengths = [len(tokens) for tokens in tokens_list]
        return {
            'tokens': tokens_list,
            'length': lengths,
        }

    ds = ds.shuffle(seed=args.seed)
    ds = ds.map(tokenize_examples, batched=True, num_proc=num_workers, remove_columns=['text'])
    train_ds = ds['train']
    total_tokens = sum(train_ds['length'])
    print('Total tokens:', total_tokens)

    shard_size = args.shard_size
    shard_idx = 0
    token_count = 0
    arr = np.full((shard_size,), dump_id, dtype=np.uint16)
    progress_bar = tqdm(desc=f'Processing shard {shard_idx}', total=shard_size, unit=' tokens')
    for i in range(len(train_ds)):
        tokens = train_ds[i]['tokens']
        assert all(token < 2**16 for token in tokens)
        cur_num_tokens = len(tokens)
        if cur_num_tokens + token_count > shard_size:
            arr[token_count:] = tokens[:shard_size - token_count]
            progress_bar.update(shard_size - token_count)
            num_remain_tokens = cur_num_tokens - (shard_size - token_count)

            # save to .npy file
            file_path = os.path.join(output_dir, f'fineweb_edu_{shard_idx:04d}.npy')
            assert all(0 <= token < 2**16 for token in arr)  # ensure that we don't miss any position in the array
            np.save(file_path, arr)
            progress_bar.write(f'Saved shard {shard_idx} to {file_path}')
            progress_bar = None

            # add remain tokens to arr for the next shard
            shard_idx += 1
            progress_bar = tqdm(desc=f'Processing shard {shard_idx}', total=shard_size, unit=' tokens')
            arr.fill(dump_id)
            arr[:num_remain_tokens] = tokens[shard_size - token_count:]
            token_count = num_remain_tokens
            progress_bar.update(num_remain_tokens)

        else:
            arr[token_count:token_count+cur_num_tokens] = tokens
            token_count += cur_num_tokens
            progress_bar.update(cur_num_tokens)

    if token_count > 1:
        file_path = os.path.join(output_dir, f'fineweb_edu_{shard_idx:04d}.npy')
        np.save(file_path, arr[:token_count])
        progress_bar.write(f'Saved shard {shard_idx} to {file_path}')

def main():
    parser = argparse.ArgumentParser(
        description='Prepare fineweb-edu dataset for pre-training GPT2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()
    prepare_fineweb_edu(args)

def add_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--seed',
        help='Seed for random number generator',
        type=int,
        default=1061109567,
    )
    parser.add_argument(
        '--tokenizer',
        help='Which tokenizer to use for tokenizing dataset (see tiktoken for all available options)',
        type=str,
        default='gpt2',
    )
    parser.add_argument(
        '--shard-size',
        help='Size of each shard to be saved to .npy file',
        type=int,
        default=int(1e8),
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory',
        type=str,
        default='./fineweb_edu_10BT',
    )
    parser.add_argument(
        '--num-workers',
        help='Number of workers',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--cache-dir',
        help='Where to cache the downloaded dataset. If `None`, use the default cache directory of the datasets library',
        type=str,
    )


if __name__ == '__main__':
    main()
