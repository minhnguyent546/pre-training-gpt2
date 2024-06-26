"""Pretraining GPT-2 with language modeling objective."""

import argparse
from contextlib import nullcontext
import numpy as np
import os
from tqdm.autonotebook import tqdm
from typing import Dict, Tuple

import wandb

from tokenizers import Tokenizer

from torch import Tensor
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.model import GPT, GPTConfig
import gpt2.utils as utils


def get_batch(data_ids, batch_size: int, seq_length: int) -> Tuple[Tensor, Tensor]:
    indices = torch.randint(high=len(data_ids) - seq_length, size=(batch_size,))
    inputs = torch.stack([torch.from_numpy(data_ids[idx:idx+seq_length]).type(torch.int32) for idx in indices])
    labels = torch.stack([torch.from_numpy(data_ids[idx+1:idx+1+seq_length]).type(torch.int64) for idx in indices])
    return inputs, labels

@torch.no_grad()
def eval_model(model: GPT | DDP, test_ids, valid_steps: int, criterion, config: dict) -> Dict[str, float]:
    is_training = model.training
    model.eval()
    accum_valid_loss = 0.0
    valid_iter = tqdm(range(valid_steps), desc='Validating model')
    for _ in valid_iter:
        inputs, labels = get_batch(test_ids, config['batch_size'], config['seq_length'])
        inputs = inputs.to(model.device)
        labels = labels.to(model.device)
        logits = model(inputs)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        accum_valid_loss += loss.item()

    accum_valid_loss /= valid_steps

    model.train(is_training)

    return {
        'loss': accum_valid_loss,
    }

def train_model(config: dict):
    utils.set_seed(config['seed'])

    data_save_path = utils.ensure_dir(config['data_save_path'])
    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # load data
    train_ids = np.fromfile(os.path.join(data_save_path, 'train.bin'), dtype=np.int16)
    test_ids = np.fromfile(os.path.join(data_save_path, 'test.bin'), dtype=np.int16)

    # load trained tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(checkpoints_dir, config['tokenizer_basename']))
    config['vocab_size'] = tokenizer.get_vocab_size()

    # logging with wandb
    wandb_run = None
    if config['wandb_logging'] and config['master_process']:
        wandb_run = wandb.init(
            project=config['project_name'],
            name=config['expr_name'],
            config=config,
            id=config['wandb_resume_id'],
            resume='must' if config['wandb_resume_id'] is not None else None,
        )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # mixed precision training with fp16
    dtype = torch.float32
    autocast_context = nullcontext()
    if config['fp16'] and torch.cuda.is_available():
        dtype = torch.float16
        autocast_context = torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

    # resume from previous checkpoint
    preload_checkpoint = config['preload_checkpoint']
    saved_states = None
    if preload_checkpoint is None:
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
    else:
        print(f'Loading states from checkpoint {preload_checkpoint}')
        saved_states = torch.load(preload_checkpoint, map_location=device)
        required_keys = [
            'model_state_dict',
            'optimizer_state_dict',
            'lr_scheduler_state_dict',
            'config'
        ]
        if scaler.is_enabled():
            required_keys.append('scaler_state_dict')
        for key in required_keys:
            if key not in saved_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        gpt_config = GPTConfig(**saved_states['config'])

    model = GPT(gpt_config)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['lr']
    optimizer = utils.make_optimizer(
        model,
        config['optim'],
        lr=learning_rate,
        weight_decay=config['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: learning_rate * utils.noam_decay(
            step,
            config['d_model'],
            config['warmup_steps']
        ),
    )

    initial_step = 0
    accum_train_loss = 0.0
    if saved_states is not None:
        model.load_state_dict(saved_states['model_state_dict'])
        optimizer.load_state_dict(saved_states['optimizer_state_dict'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler_state_dict'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states['scaler_state_dict'])
        if 'global_step' in saved_states:
            initial_step = saved_states['global_step']
        if 'accum_train_loss' in saved_states:
            accum_train_loss = saved_states['accum_train_loss']

    # convert the model to distributed data parallel
    raw_model = model
    if config['ddp']:
        model = DDP(model, device_ids=[config['local_rank']], output_device=config['local_rank'])
        raw_model = model.module

    # training loop
    train_steps = config['train_steps']
    valid_steps = config['valid_steps']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    gradient_accum_step = config['gradient_accum_step']
    model.train()

    if config['master_process']:
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'Model has {num_parameters / 10 ** 6:0.2f}M parameters')

    train_iter = tqdm(
        range(initial_step, train_steps),
        desc=f'Training model on gpu {config.get("local_rank", "")}',
        disable=config['local_rank'] != 0,
    )
    torch.cuda.empty_cache()
    batch_loss = 0.0
    global_step = initial_step
    batch_idx = 0
    while global_step < train_steps:
        optimizer.zero_grad()

        if config['ddp']:
            # we only sync gradients at the last step of gradient accumulation
            # we can use the above trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
            model.require_backward_grad_sync = (batch_idx + 1) % gradient_accum_step == 0
        with autocast_context:
            inputs, labels = get_batch(train_ids, config['batch_size'], config['seq_length'])
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if gradient_accum_step > 1:
            loss /= gradient_accum_step
        batch_loss += loss.item()

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accum_step == 0 or batch_idx + 1 == train_steps:
            if config['max_grad_norm'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

            scaler.step(optimizer)
            scaler.update()

            if wandb_run is not None:
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                    wandb_run.log({f'learning_rate/group-{group_id}': group_lr}, step=global_step)
                wandb_run.log({'loss/batch_loss': batch_loss}, step=global_step)

            lr_scheduler.step()

            train_iter.set_postfix({'loss': f'{batch_loss:0.3f}'})
            accum_train_loss += batch_loss
            batch_loss = 0.0

            if config['master_process'] and (global_step + 1) % valid_interval == 0:
                valid_results = eval_model(model, test_ids, valid_steps, criterion, config)
                if wandb_run is not None:
                    wandb_run.log({
                        'loss/train': accum_train_loss / valid_interval,
                        'loss/valid': valid_results['loss'],
                    }, step=global_step + 1)
                accum_train_loss = 0.0

            if config['master_process'] and (global_step + 1) % save_interval == 0:
                checkpoint_dict = {
                    'global_step': global_step + 1,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'config': vars(gpt_config),
                    'accum_train_loss': accum_train_loss,
                }
                if scaler.is_enabled():
                    checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
                model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step + 1}.pt')
                torch.save(checkpoint_dict, model_save_path)

            global_step += 1
            train_iter.update()

        batch_idx += 1

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

    config['rank'] = int(os.environ.get('RANK', -1))

    config['ddp'] = config['rank'] != -1
    config['master_process'] = config['rank'] in (-1, 0)
    if config['ddp']:
        config['local_rank'] = int(os.environ['LOCAL_RANK'])
        config['world_size'] = int(os.environ['WORLD_SIZE'])

        # set appropriate CUDA device
        torch.cuda.set_device(config['local_rank'])

        # init process group
        init_process_group(backend='nccl')  # nccl, gloo, etc

        # scale down the gradient accumulation step
        assert config['gradient_accum_step'] % config['world_size'] == 0
        config['gradient_accum_step'] //= config['world_size']

        # add offset for seed
        config['seed'] += config['rank']

    train_model(config)

    if config['ddp']:
        destroy_process_group()

if __name__ == '__main__':
    main()
