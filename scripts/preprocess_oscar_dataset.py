#!/usr/bin/env python3

"""Preprocessing OSCAR dataset and saving as parquet files for pre-training GPT2 model"""

import argparse
import os
from typing import Any

import datasets
from tokenizers import Tokenizer


def preprocess_dataset(
    train_files: list[str],
    validation_files: list[str],
    tokenizer: Tokenizer,
    out_dir: str,
    keep_text: bool = False,
    num_workers: int = 1,
) -> None:
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(
        'text',
        data_files={
            'train': train_files,
            'validation': validation_files,
        },
        num_proc=num_workers,
    )
    def process_examples(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            'ids': [tokenizer.encode(item).ids for item in examples['text']],
        }
    remove_columns = [] if keep_text else ['text']
    raw_dataset = raw_dataset.map(
        process_examples,
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_workers,
    )

    for split in raw_dataset:
        save_path = os.path.join(out_dir, f'{split}.parquet')
        total_tokens = sum(len(item['ids']) for item in raw_dataset[split])
        raw_dataset[split].to_parquet(save_path)
        print(f'Saved {split} split contains total {total_tokens} tokens to {save_path}')

def add_opts(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--train-files',
        help='Path to the train text files',
        required=True,
        nargs='+',
        type=str,
    )
    parser.add_argument(
        '--validation-files',
        help='Path to the validation text files',
        required=True,
        nargs='+',
        type=str,
    )
    parser.add_argument(
        '--tokenizer',
        help='Path to the tokenizer',
        type=str,
        default='./tokenizer.json',
    )
    parser.add_argument(
        '--seed',
        help='Seed for random number generation',
        type=int,
        default=1061109567,
    )
    parser.add_argument(
        '--out-dir',
        help='Output directory',
        type=str,
        default='./oscar-vi',
    )
    parser.add_argument(
        '--keep-text',
        help='Whether to keep text field after encoding',
        action='store_true',
    )
    parser.add_argument(
        '--num-workers',
        help='Number of workers',
        type=int,
        default=1,
    )

def main():
    parser = argparse.ArgumentParser(
        'Preprocess OSCAR dataset for pre-training GPT2 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    preprocess_dataset(
        args.train_files,
        args.validation_files,
        tokenizer,
        args.out_dir,
        keep_text=args.keep_text,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
