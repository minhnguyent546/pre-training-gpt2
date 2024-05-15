"""
prepare dataset and tokenizer for training
"""

import argparse
import os
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import tokenizers
from tokenizers import Tokenizer, AddedToken
import tokenizers.models
import tokenizers.decoders
import tokenizers.pre_tokenizers
import tokenizers.trainers

import utils


def build_tokenizer(
    data_iter,
    vocab_size: int = 30_000,
    min_freq: int = 1,
    show_progress: bool = True
) -> Tokenizer:
    tokenizer = Tokenizer(tokenizers.models.WordPiece(
        unk_token='[UNK]',
        max_input_chars_per_word=100,
    ))  # pyright: ignore[reportCallIssue]
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.decoder = tokenizers.decoders.WordPiece(
        prefix='##',
        cleanup=False,
    )
    trainer = tokenizers.trainers.WordPieceTrainer(
        vocab_size=vocab_size - 1,
        min_frequency=min_freq,
        show_progress=show_progress,
        special_tokens=['[UNK]'],
        continuing_subword_prefix='##'
    )
    tokenizer.train_from_iterator(data_iter, trainer=trainer)
    tokenizer.add_special_tokens([AddedToken("\n")])
    return tokenizer

def preprocess(config: dict):
    utils.set_seed(config['seed'])

    data_file_path = config['data_file_path']
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    tokenizer = build_tokenizer(utils.chunks(data), vocab_size=config['vocab_size'])
    print(f'Vocab size: {tokenizer.get_vocab_size()}')

    test_size = config.get('test_size', 0.1)
    train_data_size = int(len(data) * (1 - test_size))
    train_data = data[:train_data_size]
    test_data = data[train_data_size:]

    print('Encoding data')
    # this may not be an efficient way to encode data
    train_ids = tokenizer.encode(train_data).ids
    test_ids = tokenizer.encode(test_data).ids

    print('save to bin files')
    train_ids = np.array(train_ids, dtype=np.int16)
    test_ids = np.array(test_ids, dtype=np.int16)

    # save as bin files
    data_save_path = utils.ensure_dir(config['data_save_path'])
    train_ids.tofile(os.path.join(data_save_path, 'train.bin'))
    test_ids.tofile(os.path.join(data_save_path, 'test.bin'))

    # save tokenizer
    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])
    tokenizer_save_path = os.path.join(checkpoints_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_save_path)

def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset and tokenizer for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        '--config',
        help='Path to the config file',
        default='./config/config.yaml',
    )
    args = parser.parse_args()
    config = utils.load_yaml_config(args.config)
    preprocess(config)


if __name__ == '__main__':
    main()
