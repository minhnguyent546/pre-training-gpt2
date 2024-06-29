#!/usr/bin/env python3

"""
Downloading and slicing (if necessary) the OSCAR dataset (https://huggingface.co/datasets/oscar-corpus/oscar) from Huggingface
"""

import argparse
import os

import datasets
from datasets.utils.info_utils import VerificationMode


def add_opts(parser: argparse.ArgumentParser):
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
        '--split',
        help='Which split to download',
        type=str,
        default='unshuffled_original_vi',
    )
    parser.add_argument(
        '--verify-data',
        help='Whether to verify the downloaded data',
        action='store_true',
    )
    parser.add_argument(
        '--val-size',
        help='Size of the validation set',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--num-workers',
        help='Number of workers',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--max-num-docs',
        help='Maximum number of documents',
        type=int,
    )

def main():
    parser = argparse.ArgumentParser(
        'Prepare the OSCAR dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    raw_dataset: datasets.Dataset = datasets.load_dataset(
        'oscar',
        args.split,
        split=f'train[:{args.max_num_docs}]' if args.max_num_docs else 'train',
        num_proc=args.num_workers,
        trust_remote_code=True,
        verification_mode=VerificationMode.BASIC_CHECKS if args.verify_data else VerificationMode.NO_CHECKS,
    )
    val_size = args.val_size
    if val_size > 1:
        val_size = int(val_size)
    raw_dataset = raw_dataset.train_test_split(test_size=val_size, seed=args.seed)
    raw_dataset['validation'] = raw_dataset.pop('test')

    os.makedirs(args.out_dir, exist_ok=True)
    train_file_path = os.path.join(args.out_dir, 'train.txt')
    validation_file_path = os.path.join(args.out_dir, 'validation.txt')

    with open(train_file_path, 'w', encoding='utf-8') as f:
        f.writelines(raw_dataset['train']['text'])
    print(f'Wrote {len(raw_dataset["train"])} documents to {train_file_path}')

    with open(validation_file_path, 'w', encoding='utf-8') as f:
        f.writelines(raw_dataset['validation']['text'])
    print(f'Wrote {len(raw_dataset["validation"])} documents to {validation_file_path}')


if __name__ == '__main__':
    main()
