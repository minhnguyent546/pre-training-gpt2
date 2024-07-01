#!/usr/bin/env python3

"""Preprocessing OSCAR dataset and saving as binary files for pre-training GPT2 model"""

import argparse
import os
import regex
import unicodedata
from typing import Any

import numpy as np

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
        examples['text'] = [
            clean_text(text, strip=True, keep_punct=True)
            for text in examples['text']
        ]
        ids_list = tokenizer.encode_batch(examples['text'])
        ids_list = [ids.ids + [tokenizer.token_to_id('<|endoftext|>')] for ids in ids_list]
        return {
            'ids': ids_list,
        }

    remove_columns = [] if keep_text else ['text']
    raw_dataset = raw_dataset.map(
        process_examples,
        batched=True,
        remove_columns=remove_columns,
        num_proc=num_workers,
    )
    for split in raw_dataset:
        save_path = os.path.join(out_dir, f'{split}.dat')
        total_tokens = sum(len(item['ids']) for item in raw_dataset[split])
        arr = np.memmap(save_path, mode='w+', dtype=np.uint16, shape=(total_tokens,))
        num_shards = 1024
        ptr = 0
        for idx in range(num_shards):
            shard = raw_dataset[split].shard(num_shards, index=idx, contiguous=True)
            ids_list = np.concatenate(shard['ids'])
            arr[ptr:ptr+len(ids_list)] = ids_list
            ptr += len(ids_list)
        arr.flush()
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

# copied from ~gpt2/utils.py
def clean_text(text: str, *, strip: bool = True, keep_punct: bool = True) -> str:
    # NFC normalization
    text = unicodedata.normalize('NFC', text)
    # remove non-latin characters (but keep numbers, punctuations, and whitespaces)
    if keep_punct:
        text = regex.sub(r'([^\p{Latin}\p{Punctuation}0-9\s]+)', r'', text)
    else:
        text = regex.sub(r'([^\p{Latin}0-9\s]+)', r'', text)
    # normalize tone
    text = normalize_tone(text)
    if strip:
        text = text.strip()
    return text


tone_normalization_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
}

def normalize_tone(text: str) -> str:
    """
    Tone normalization for Vietnamese (source: https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md)
    """
    for orig, repl in tone_normalization_map.items():
        text = text.replace(orig, repl)
    return text


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
