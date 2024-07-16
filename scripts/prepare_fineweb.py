#!/usr/bin/env python

"""
Download and tokenize the fineweb dataset (10BT subset) (https://huggingface.co/datasets/HuggingFaceFW/fineweb)
for pre-training GPT2. Adapted from https://github.com/karpathy/build-nanogpt.

This script will need approximately 80GiB space on disk to download and process
the dataset, plus 20GiB for tokens files.
"""

import argparse
import multiprocessing as mp
import os
import random
from functools import partial
from typing import Any

import datasets
import numpy as np
import tiktoken
from tqdm.autonotebook import tqdm


def prepare_fineweb_edu(args: argparse.Namespace) -> None:
    UINT16_LIMIT = 2 ** 16 - 1

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # tokenizer
    tokenizer = tiktoken.encoding_for_model(args.tokenizer)
    dump_id = tokenizer.n_vocab
    assert dump_id <= UINT16_LIMIT

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_workers = args.num_workers

    ds = datasets.load_dataset(
        'HuggingFaceFW/fineweb-edu' if args.fineweb_edu else 'HuggingFaceFW/fineweb',
        'sample-10BT',
        num_proc=None if args.streaming else args.num_workers,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        split='train',
    )

    shard_size = args.shard_size
    shard_idx = 0
    token_count = 0
    arr = np.full((shard_size,), dump_id, dtype=np.uint16)
    num_approx_shards = 10 * 10**9 // shard_size
    print(f'Approximately {num_approx_shards} shards will be created with size {shard_size} tokens each')
    shard_file_basename = 'fineweb_edu' if args.fineweb_edu else 'fineweb'
    progress_bar = tqdm(desc=f'Processing shard {shard_idx}', total=shard_size, unit=' tokens')
    with mp.Pool(num_workers) as pool:
        for item in pool.imap(partial(tokenize_example, tokenizer=tokenizer), ds, chunksize=128):
            tokens = item['tokens']
            cur_num_tokens = len(tokens)
            if cur_num_tokens + token_count > shard_size:
                arr[token_count:] = tokens[:shard_size - token_count]
                progress_bar.update(shard_size - token_count)
                num_remain_tokens = cur_num_tokens - (shard_size - token_count)

                # save to .npy file
                file_path = os.path.join(output_dir, f'{shard_file_basename}_{shard_idx:04d}.npy')
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
            file_path = os.path.join(output_dir, f'{shard_file_basename}_{shard_idx:04d}.npy')
            np.save(file_path, arr[:token_count])
            progress_bar.write(f'Saved shard {shard_idx} to {file_path}')

def tokenize_example(example: dict[str, Any], tokenizer: tiktoken.Encoding) -> dict[str, Any]:
    tokens = tokenizer.encode_ordinary(example['text'])
    tokens.append(tokenizer.eot_token)
    return {
        'tokens': tokens,
        'length': len(tokens),
    }

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
        '--fineweb-edu',
        help='Whether to use the edu version of fineweb dataset (fineweb-edu)',
        action='store_true',
    )
    parser.add_argument(
        '--streaming',
        help='Stream the dataset instead of downloading',
        action='store_true',
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
