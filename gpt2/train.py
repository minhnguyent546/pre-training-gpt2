import argparse
from contextlib import nullcontext
import os
import numpy as np
from tqdm.autonotebook import tqdm
from typing import Tuple, Dict

from tokenizers import Tokenizer

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import utils
from model import GPT, GPTConfig


def get_batch(data_ids, batch_size: int, seq_length: int) -> Tuple[Tensor, Tensor]:
    indices = torch.randint(high=len(data_ids) - seq_length, size=(batch_size,))
    inputs = torch.stack([torch.from_numpy(data_ids[idx:idx+seq_length]).type(torch.int32) for idx in indices])
    labels = torch.stack([torch.from_numpy(data_ids[idx+1:idx+1+seq_length]).type(torch.int64) for idx in indices])
    return inputs, labels

@torch.no_grad()
def eval_model(model: GPT, test_ids, valid_steps: int, criterion, config: dict) -> Dict[str, float]:
    is_training = model.training
    model.eval()
    accum_valid_loss = 0.0
    valid_iter = tqdm(range(valid_steps), desc='Validating model')
    for _ in valid_iter:
        inputs, labels = get_batch(test_ids, config['batch_size'], config['seq_length'])
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        accum_valid_loss += loss.item()

    accum_valid_loss /= valid_steps

    if is_training:
        model.train()

    return {
        'loss': accum_valid_loss,
    }

def train_model(config: dict):
    utils.set_seed(config['seed'])

    data_save_path = utils.ensure_dir(config['data_save_path'])
    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # load data
    train_ids = np.fromfile(os.path.join(data_save_path, 'train.bin'), dtype=np.int16)
    test_ids = np.fromfile(os.path.join(data_save_path, 'train.bin'), dtype=np.int16)

    # load trained tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(checkpoints_dir, config['tokenizer_basename']))

    # tensorboard
    writer = SummaryWriter(config['expr_name'])

    # model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    gpt_config = GPTConfig(
        vocab_size=config['vocab_size'],
        seq_length=config['seq_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        activation=config['activation'],
    )
    model = GPT(gpt_config)
    model.to(device)

    # mixed precision training with fp16
    dtype = torch.float32
    autocast_context = nullcontext()
    if config['fp16'] and torch.cuda.is_available():
        dtype = torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    criterion = nn.CrossEntropyLoss()
    learning_rate = config['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate * utils.noam_decay(step, config['d_model'], config['warmup_steps']),
    )

    # resume from previous checkpoint
    initial_step = 0
    accum_train_loss = 0.0
    preload_checkpoint = config['preload_checkpoint']
    if preload_checkpoint is not None:
        saved_states = torch.load(preload_checkpoint, map_location=device)
        initial_step = saved_states['global_step']
        model.load_state_dict(saved_states['model_state_dict'])
        optimizer.load_state_dict(saved_states['optimizer_state_dict'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler_state_dict'])
        gpt_config = GPTConfig(saved_states['gpt_config'])
        accum_train_loss = saved_states['accum_train_loss']

    # training loop
    train_steps = config['train_steps']
    valid_steps = config['valid_steps']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    model.train()

    train_iter = tqdm(range(initial_step, train_steps), desc='Training model')
    for global_step in train_iter:
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        with autocast_context:
            inputs, labels = get_batch(train_ids, config['batch_size'], config['seq_length'])
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        scaler.scale(loss).backward()

        if config['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

        scaler.step(optimizer)
        scaler.update()

        for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
            writer.add_scalar(f'learning_rate/group-{group_id}', group_lr, global_step)

        lr_scheduler.step()

        train_iter.set_postfix({'loss': loss.item()})
        accum_train_loss += loss.item()

        writer.add_scalar('loss/batch_loss', loss.item(), global_step)
        writer.flush()

        if (global_step + 1) % valid_interval == 0:
            valid_results = eval_model(model, test_ids, valid_steps, criterion, config)
            writer.add_scalars('loss', {
                'train': accum_train_loss / valid_interval,
                'valid': valid_results['loss'],
            }, global_step + 1)
            writer.flush()
            accum_train_loss = 0.0

        if (global_step + 1) % save_interval:
            checkpoint_dict = {
                'global_step': global_step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'gpt_config': vars(gpt_config),
                'accum_train_loss': accum_train_loss,
            }
            model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step}.pt')
            torch.save(checkpoint_dict, model_save_path)

def main():
    parser = argparse.ArgumentParser(
        description='Train GPT2 model',
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
    train_model(config)


if __name__ == '__main__':
    main()
