"""Pretraining GPT-2 with language modeling objective."""

import argparse
from contextlib import nullcontext
import os
from tqdm.autonotebook import tqdm
from typing import Dict, Any, Literal, Tuple

import wandb

import numpy as np

from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2.model import GPT, GPTConfig
import gpt2.utils as utils


def train_model(config: dict[str, Any]):
    utils.set_seed(config['seed'])

    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # load trained tokenizer
    tokenizer: Tokenizer = Tokenizer.from_file(config['tokenizer'])
    config['vocab_size'] = tokenizer.get_vocab_size()

    # train and validation data
    train_data = np.memmap(config['train_data'], mode='r', dtype=np.uint16)
    validation_data = np.memmap(config['validation_data'], mode='r', dtype=np.uint16)

    # logging with wandb
    wandb_run = None
    wandb_config = config['wandb']
    if config['master_process'] and wandb_config['logging']:
        wandb_run = wandb.init(
            project=wandb_config['project'],
            name=wandb_config['name'],
            config=config,
            tags=wandb_config.get('tags', None),
            notes=wandb_config.get('notes', None),
            id=wandb_config.get('resume_id', None),
            resume='must' if wandb_config.get('resume_id', None) is not None else None,
        )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # mixed precision training with fp16
    dtype = torch.float32
    autocast_context = nullcontext()
    if config['fp16'] and torch.cuda.is_available():
        if config['master_process']:
            print('Training with fp16 precision')
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
        if config['master_process']:
            print(f'Loading states from checkpoint {preload_checkpoint}')
        saved_states = torch.load(preload_checkpoint, map_location=device)
        required_keys = [
            'model',
            'optimizer',
            'lr_scheduler',
            'config'
        ]
        if scaler.is_enabled():
            required_keys.append('scaler')
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
    running_loss = AverageMeter('running_loss', device=device)
    if saved_states is not None:
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optimizer'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states['scaler'])
        if 'global_step' in saved_states:
            initial_step = saved_states['global_step']
        if 'running_loss' in saved_states:
            if config['ddp']:
                running_loss = AverageMeter(**saved_states['running_loss'][config['rank']])
            else:
                running_loss = AverageMeter(**saved_states['running_loss'])

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

    if config['ddp']:
        train_iter = tqdm(
            range(initial_step, train_steps),
            desc=f'Training model on rank {config["rank"]}',
            disable=config['local_rank'] != 0,
        )
    else:
        train_iter = tqdm(range(initial_step, train_steps), desc='Training model')

    batch_loss = 0.0
    batch_idx = 0
    global_step = initial_step
    while global_step < train_steps:
        torch.cuda.empty_cache()

        input_ids, labels = get_batch(
            train_data,
            config['train_batch_size'],
            config['seq_length'],
            device=device,
        )

        optimizer.zero_grad()

        if config['ddp']:
            # we only sync gradients at the last step of gradient accumulation
            # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
            model.require_backward_grad_sync = (batch_idx + 1) % gradient_accum_step == 0
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if gradient_accum_step > 1:
            loss /= gradient_accum_step
        batch_loss += loss.item()

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accum_step == 0:
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
            running_loss.update(batch_loss)
            batch_loss = 0.0

            if config['master_process'] and (global_step + 1) % valid_interval == 0:
                valid_results = eval_model(model, criterion, validation_data, config, valid_steps=valid_steps)
                if config['ddp']:
                    running_loss.reduce(dst=config['rank'])
                if wandb_run is not None:
                    wandb_run.log({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                    }, step=global_step + 1)
                running_loss.reset()

            if config['master_process'] and (global_step + 1) % save_interval == 0:
                if config['ddp']:
                    running_losses = [None for _ in range(config['world_size'])] if config['master_process'] else None
                    dist.gather_object(vars(running_loss), running_losses, dst=config['rank'])
                else:
                    running_losses = running_loss
                checkpoint_dict = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': vars(gpt_config),
                    'global_step': global_step + 1,
                    'running_loss': running_losses,
                }
                if scaler.is_enabled():
                    checkpoint_dict['scaler'] = scaler.state_dict()
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

    setup_ddp(config)

    train_model(config)

    if config['ddp']:
        dist.destroy_process_group()

def get_batch(
    data_ids,
    batch_size: int,
    seq_length: int,
    device: torch.device | None = None,
) -> Tuple[Tensor, Tensor]:
    indices = torch.randint(high=len(data_ids) - seq_length, size=(batch_size,))
    input_ids = torch.stack([torch.from_numpy(data_ids[idx:idx+seq_length].copy().astype(np.int32)) for idx in indices])
    labels = torch.stack([torch.from_numpy(data_ids[idx+1:idx+1+seq_length].copy().astype(np.int64)) for idx in indices])
    if device is not None:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
    return input_ids, labels

@torch.no_grad()
def eval_model(
    model: GPT | DDP,
    criterion,
    eval_data,
    config: dict[str, Any],
    valid_steps: int | None = None,
) -> Dict[str, float]:
    device = model.device
    evaluation_loss = AverageMeter('evaluation_loss', device=device)

    if valid_steps is None:  # actually `valid_steps` should be provided
        valid_steps = len(eval_data)
    else:
        if config['ddp']:
            assert valid_steps % config['world_size'] == 0, \
                f'`valid_steps` must be divisible by `world_size` ({config["world_size"]})'
            valid_steps //= config['world_size']

    if config['ddp']:
        valid_iter = tqdm(
            valid_steps,
            total=valid_steps,
            desc=f'Evaluating model on rank {config["rank"]}',
            disable=config['local_rank'] != 0,
        )
    else:
        valid_iter = tqdm(range(valid_steps), total=valid_steps, desc='Evaluating model')

    is_training = model.training
    model.eval()

    for batch_idx in valid_iter:
        input_ids, labels = get_batch(
            eval_data,
            config['train_batch_size'],
            config['seq_length'],
            device=device,
        )

        # TODO: consider using fp16 for evaluation
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        evaluation_loss.update(loss.item())
        valid_iter.set_postfix({'loss': f'{loss.item():0.3f}'})

        if batch_idx + 1 >= valid_steps:
            break

    if config['ddp']:
        evaluation_loss.reduce(dst=config['rank'])

    model.train(is_training)

    return {
        'loss': evaluation_loss.average,
    }

def setup_ddp(config: dict[str, Any]) -> None:
    config['rank'] = int(os.environ.get('RANK', -1))
    config['ddp'] = config['rank'] != -1
    config['master_process'] = config['rank'] in (-1, 0)
    if config['ddp']:
        config['local_rank'] = int(os.environ['LOCAL_RANK'])
        config['world_size'] = int(os.environ['WORLD_SIZE'])

        # set appropriate CUDA device
        torch.cuda.set_device(config['local_rank'])

        # init process group
        dist.init_process_group(backend=config.get('ddp_backend', 'nccl'))  # nccl, gloo, etc

        # we need to divide the batch_size for batching across multiple-GPUs manually
        for key in ('train_batch_size', 'eval_batch_size'):
            assert config[key] % config['world_size'] == 0
            config[key] //= config['world_size']
            if config['master_process']:
                print(f'{key} per GPU is {config[key]}')

        # add offset for seed
        config['seed'] += config['rank']

class AverageMeter:
    """A class for working with average meters."""
    def __init__(
        self,
        name: str,
        value: int | float = 0.0,
        count: int = 0,
        sum: int | float = 0.0,
        device: torch.device | Literal['auto'] = 'auto',
    ) -> None:
        if count == 0:
            value = 0
            sum = 0
        self.name = name
        self.value = value
        self.count = count
        self.sum = sum
        if device == 'auto':
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def update(self, value: int | float, nums: int = 1) -> None:
        self.value = value
        self.sum += value * nums
        self.count += nums

    def reduce(self, dst: int) -> None:
        meters_to_reduce = torch.Tensor([self.sum, self.count], device=self.device).to(torch.float32)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        meters_to_reduce = torch.Tensor([self.sum, self.count], device=self.device).to(torch.float32)
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def __str__(self) -> str:
        return (
            f'{self.name}(value={self.value}, '
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}, '
            f'device={self.device})'
        )


if __name__ == '__main__':
    main()
