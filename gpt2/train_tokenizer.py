"""Training tokenizer from text files"""

import argparse
import multiprocessing as mp
from functools import partial

import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.normalizers
import tokenizers.pre_tokenizers
import tokenizers.trainers
from tokenizers import AddedToken, Tokenizer

import gpt2.utils as utils


def train_tokenizer(
    data_iter,
    vocab_size: int = 32_000,
    min_freq: int = 2,
    lowercase: bool = False,
    show_progress: bool = True,
) -> Tokenizer:
    tokenizer = Tokenizer(tokenizers.models.WordPiece(
        unk_token='<unk>',
        max_input_chars_per_word=100,
    ))  # pyright: ignore[reportCallIssue]
    # pre-tokenizer
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    # normalizers
    normalizer_list = []
    if lowercase:
        normalizer_list.append(tokenizers.normalizers.Lowercase())
    if normalizer_list:
        tokenizer.normalizer = tokenizers.normalizers.Sequence(normalizer_list)

    # decoder
    tokenizer.decoder = tokenizers.decoders.WordPiece(
        prefix='##',
        cleanup=False,
    )
    trainer = tokenizers.trainers.WordPieceTrainer(
        vocab_size=vocab_size - 1,
        min_frequency=min_freq,
        show_progress=show_progress,
        special_tokens=['<unk>', '<|endoftext|>'],
        continuing_subword_prefix='##'
    )
    tokenizer.train_from_iterator(data_iter, trainer=trainer)
    tokenizer.add_special_tokens([AddedToken("\n")])
    return tokenizer

def build_tokenizer(
    data_files: str | list[str],
    vocab_size: int,
    min_freq: int = 2,
    lowercase: bool = False,
    save_path: str | None = None,
    nun_workers: int = 1,
) -> Tokenizer:
    if isinstance(data_files, str):
        data_files = [data_files]
    data = []
    with mp.Pool(nun_workers) as pool:
        for data_file in data_files:
            with open(data_file, 'r', encoding='utf-8') as f:
                content = pool.map(partial(utils.clean_text, strip=True, keep_punct=True), f)
                data.extend(content)

    tokenizer = train_tokenizer(
        utils.chunks(data, chunk_size=10_000),
        vocab_size=vocab_size,
        min_freq=min_freq,
        lowercase=lowercase,
    )
    print(f'Vocab size: {tokenizer.get_vocab_size()}')

    if save_path is not None:
        tokenizer.save(save_path)
        print(f'Tokenizer saved to {save_path}')
    return tokenizer

def add_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Train tokenizer')
    group.add_argument(
        '--data-files',
        nargs='+',
        required=True,
        help='Path to the text files contain documents',
        type=str,
    )
    group.add_argument(
        '--output',
        help='Path to save the trained tokenizer',
        type=str,
        default='./tokenizer.json',
    )
    group.add_argument(
        '--vocab-size',
        help='Vocabulary size limit',
        type=int,
        default=32_000,
    )
    group.add_argument(
        '--min-freq',
        help='Minimum frequency of a token to be included in the vocabulary',
        type=int,
        default=3,
    )
    group.add_argument(
        '--lowercase',
        help='Whether to lowercase the text before training tokenizer',
        action='store_true',
    )
    group.add_argument(
        '--num-workers',
        help='Number of workers',
        type=int,
        default=1,
    )

def main():
    parser = argparse.ArgumentParser(
        description='Training tokenizer for GPT2 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_opts(parser)
    args = parser.parse_args()

    build_tokenizer(
        args.data_files,
        args.vocab_size,
        min_freq=args.min_freq,
        lowercase=args.lowercase,
        save_path=args.output,
    )


if __name__ == '__main__':
    main()
