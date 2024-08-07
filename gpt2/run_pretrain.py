"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
import time
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any, Literal

import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.model import GPT, GPTConfig


def train_model(config: dict[str, Any]):
    utils.set_seed(config['seed'])
    matmul_precision = config.get('matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    if config['is_master']:
        print(f'Set float32 matmul precision to {matmul_precision}')

    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # dataset
    train_dataset = LMDataset(
        config['train_dir'],
        config['train_batch_size'],
        config['seq_length'],
        shuffle=True,
        num_replicas=config['world_size'] if config['ddp'] else None,
        rank=config['rank'] if config['ddp'] else None,
    )
    validation_dataset = LMDataset(
        config['valid_dir'],
        config['eval_batch_size'],
        config['seq_length'],
        shuffle=False,
        num_replicas=config['world_size'] if config['ddp'] else None,
        rank=config['rank'] if config['ddp'] else None,
    )

    # logging with wandb
    wandb_run = None
    wandb_config = config['wandb']
    if config['is_master'] and wandb_config['logging']:
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

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == 'cuda' and config['mixed_precision'] == 'fp16':
        mp_dtype = torch.float16
        if config['is_master']:
            print('Mixed precision training is enabled with fp16')
    elif device.type == 'cuda' and config['mixed_precision'] == 'bf16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
            if config['is_master']:
                print('Mixed precision training is enabled with bf16')
        else:
            mp_dtype = torch.float16
            if config['is_master']:
                print('bf16 is not supported on your hardware, fallback to mixed precision training with fp16')
    autocast_context = torch.cuda.amp.autocast(enabled=(mp_dtype in (torch.float16, torch.bfloat16)), dtype=mp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_dtype == torch.float16))

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
            tie_weights=config['tie_weights'],
        )
    else:
        if config['is_master']:
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
    learning_rate = config['optim']['lr']
    optimizer = utils.make_optimizer(
        model,
        config['optim']['type'],
        lr=learning_rate,
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay'],
        device=device,
    )
    if config['scheduler']['decay_method'] == 'noam':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step, config['d_model'],
                config['scheduler']['warmup_steps'],
            ),
        )
    elif config['scheduler']['decay_method'] == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step, learning_rate, config['scheduler']['min_lr'],
                config['scheduler']['warmup_steps'],
                config['scheduler']['decay_steps'], factor=1/learning_rate,
            ),
        )
    else:
        raise ValueError(f'Unsupported scheduler decay method: {config["scheduler"]["decay_method"]}')

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

    raw_model = model
    # compile the model
    if config['compile']:
        if config['is_master']:
            print('Compiling the model')
        model = torch.compile(model)

    # convert the model to distributed data parallel
    if config['ddp']:
        model = DDP(model, device_ids=[config['local_rank']], output_device=config['local_rank'])

    # training loop
    train_steps = config['train_steps']
    valid_steps = config['valid_steps']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    gradient_accum_step = config['gradient_accum_step']
    num_tokens_per_batch = config['train_batch_size'] * gradient_accum_step * config['seq_length']

    model.train()

    if config['is_master']:
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'Model has {num_parameters / 10 ** 6:0.2f}M parameters')

    if config['ddp']:
        train_iter = tqdm(
            range(initial_step, train_steps),
            desc=f'GPU{config["rank"]} - Training model',
            disable=config['local_rank'] != 0,
            ncols=120,
        )
    else:
        train_iter = tqdm(
            range(initial_step, train_steps),
            desc='Training model',
            ncols=120,
        )

    batch_loss = 0.0
    batch_idx = 0
    batch_fb_time = 0.0  # batch forward + backward time
    global_step = initial_step
    torch.cuda.empty_cache()
    while global_step < train_steps:
        optimizer.zero_grad()

        ts = time.monotonic()
        input_ids, labels = train_dataset.next_batch()
        input_ids = input_ids.to(device)
        labels = labels.to(device).to(torch.int64)

        if config['ddp']:
            # we only sync gradients at the last step of gradient accumulation
            # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
            model.require_backward_grad_sync = (batch_idx + 1) % gradient_accum_step == 0
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if gradient_accum_step > 1:
            loss /= gradient_accum_step
        batch_loss += loss.detach()

        scaler.scale(loss).backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        batch_fb_time += time.monotonic() - ts

        if (batch_idx + 1) % gradient_accum_step == 0:
            if config['max_grad_norm'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

            scaler.step(optimizer)
            scaler.update()

            batch_throughput = num_tokens_per_batch / batch_fb_time
            if config['ddp']:
                batch_throughput *= config['world_size']  # estimate throughput when training with multiple gpus

            if wandb_run is not None:
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                    wandb_run.log({f'learning_rate/group-{group_id}': group_lr}, step=global_step)
                wandb_run.log({
                    'loss/batch_loss': batch_loss,
                    'throughput': batch_throughput,
                }, step=global_step)

            lr_scheduler.step()

            running_loss.update(batch_loss)
            train_iter.set_postfix({
                'loss': f'{batch_loss:0.3f}',
                'throughput': f'{batch_throughput:0.3f} tokens/s'
            })
            batch_loss = 0.0
            batch_fb_time = 0.0

            if (global_step + 1) % valid_interval == 0:
                if config['ddp']:
                    running_loss.reduce(dst=config['master_rank'])
                valid_results = eval_model(
                    model,
                    criterion,
                    validation_dataset,
                    valid_steps,
                    config,
                    autocast_context,
                )
                if wandb_run is not None:
                    wandb_run.log({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                    }, step=global_step + 1)
                running_loss.reset()

            if (global_step + 1) % save_interval == 0:
                if config['ddp']:
                    running_losses = [None for _ in range(config['world_size'])] if config['is_master'] else None
                    dist.gather_object(vars(running_loss), running_losses, dst=config['master_rank'])
                else:
                    running_losses = running_loss
                if config['is_master']:
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
                    utils.ensure_num_saved_checkpoints(
                        config['checkpoints_dir'],
                        'gpt2',
                        config['saved_checkpoint_limit'],
                    )
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

@torch.no_grad()
def eval_model(
    model: GPT | DDP,
    criterion,
    eval_dataset: LMDataset,
    valid_steps: int,
    config: dict[str, Any],
    autocast_context=nullcontext,
) -> dict[str, float]:
    device = model.device
    evaluation_loss = AverageMeter('evaluation_loss', device=device)

    if config['ddp']:
        valid_iter = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc=f'GPU{config["rank"]} - Evaluating model',
            disable=config['local_rank'] != 0,
            ncols=120,
        )
    else:
        valid_iter = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc='Evaluating model',
            ncols=120,
        )

    is_training = model.training
    model.eval()

    for batch_idx in valid_iter:
        input_ids, labels = eval_dataset.next_batch()
        input_ids = input_ids.to(device)
        labels = labels.type(torch.int64).to(device)

        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        evaluation_loss.update(loss)
        valid_iter.set_postfix({'loss': f'{loss:0.3f}'})

        if batch_idx + 1 >= valid_steps:
            break

    if config['ddp']:
        evaluation_loss.reduce(dst=config['master_rank'])

    model.train(is_training)

    return {
        'loss': evaluation_loss.average,
    }

def setup_ddp(config: dict[str, Any]) -> None:
    config['rank'] = int(os.environ.get('RANK', -1))
    config['ddp'] = config['rank'] != -1
    config['master_rank'] = 0 if config['ddp'] else -1
    config['is_master'] = config['rank'] == config['master_rank']
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
            if config['is_master']:
                if key == 'train_batch_size' and config['gradient_accum_step'] > 1:
                    print(
                        f'{key} per GPU is {config[key]} '
                        f'(with gradient accum steps = {config["gradient_accum_step"]})'
                    )
                else:
                    print(f'{key} per GPU is {config[key]}')

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
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
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

    def __repr__(self) -> str:
        return (
            f'{self.name}(value={self.value}, '
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}, '
            f'device={self.device})'
        )


if __name__ == '__main__':
    main()
