"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
import time
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any

import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.meters import AverageMeter
from gpt2.model import GPT, GPTConfig


def train_model(config: dict[str, Any]):
    utils.set_seed(config['seed'])
    matmul_precision = config.get('matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    if config['is_master']:
        print(f'Set float32 matmul precision to {matmul_precision}')

    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    train_batch_size = config['train_batch_size']
    eval_batch_size = config['eval_batch_size']

    if train_batch_size % config['world_size'] != 0:
        raise ValueError('train_batch_size must be divisible by world_size')
    if eval_batch_size % config['world_size'] != 0:
        raise ValueError('eval_batch_size must be divisible by world_size')
    train_batch_size //= config['world_size']
    eval_batch_size //= config['world_size']
    effective_batch_size = train_batch_size * config['world_size'] * config['gradient_accum_step']
    if config['is_master']:
        print(
            f'Effective batch size: {effective_batch_size} '
            f'(micro_batch_size={train_batch_size}, '
            f'gradient_accum_step={config["gradient_accum_step"]}, '
            f'num_devices={config["world_size"]})'
        )

    # dataset
    train_dataset = LMDataset(
        config['train_dir'],
        train_batch_size,
        config['seq_length'],
        num_replicas=config['world_size'],
        rank=config['rank'],
    )
    validation_dataset = LMDataset(
        config['valid_dir'],
        eval_batch_size,
        config['seq_length'],
        num_replicas=config['world_size'],
        rank=config['rank'],
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
                print('bf16 is not supported on your hardware, fallback to fp16')
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
            tie_weights=False,
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
        config['tie_weights'] = config['tie_weights'] & gpt_config.tie_weights
    model = GPT(gpt_config)
    model.to(device)
    if config['tie_weights']:
        gpt_config.tie_weights = config['tie_weights']
        model.use_tied_weights = True
        model.tie_weights()
    criterion = nn.CrossEntropyLoss()
    learning_rate = config['optim']['lr']
    optimizer = utils.make_optimizer(
        model,
        device,
        config['optim']['type'],
        lr=learning_rate,
        betas=config['optim']['betas'],
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay'],
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
    num_tokens_per_batch = train_batch_size * gradient_accum_step * config['seq_length']

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
    batch_fb_time = 0.0  # batch forward + backward time
    global_step = initial_step

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    while global_step < train_steps:
        for batch_idx, (input_ids, labels) in enumerate(train_dataset):
            ts = time.perf_counter()
            input_ids = input_ids.to(device)
            labels = labels.to(device)

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
            batch_fb_time += time.perf_counter() - ts

            if (batch_idx + 1) % gradient_accum_step == 0:
                if config['max_grad_norm'] > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                batch_throughput = num_tokens_per_batch / batch_fb_time
                batch_throughput *= config['world_size']  # estimate throughput across devices

                if wandb_run is not None:
                    for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                        wandb_run.log({f'learning_rate/group_{group_id}': group_lr}, step=global_step)
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
                        running_losses = running_loss.gather_object(
                            dst=config['master_rank'],
                            world_size=config['world_size'],
                            is_master=config['is_master'],
                        )
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
                if global_step >= train_steps:
                    break

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

    cleanup_ddp(config)

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
        progress_bar = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc=f'GPU{config["rank"]} - Evaluating model',
            disable=config['local_rank'] != 0,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc='Evaluating model',
            ncols=120,
        )

    # set model in evaluation mode
    is_training = model.training
    model.eval()

    for batch_idx, (input_ids, labels) in enumerate(eval_dataset):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        evaluation_loss.update(loss.detach())
        progress_bar.set_postfix({'loss': f'{loss:0.3f}'})
        progress_bar.update()
        if (batch_idx + 1) >= valid_steps:
            break

    # set model back to the original mode
    model.train(is_training)

    if config['ddp']:
        evaluation_loss.reduce(dst=config['master_rank'])

    return {
        'loss': evaluation_loss.average,
    }

def setup_ddp(config: dict[str, Any]) -> None:
    config['rank'] = int(os.environ.get('RANK', 0))
    config['local_rank'] = int(os.environ.get('LOCAL_RANK', 0))
    config['world_size'] = int(os.environ.get('WORLD_SIZE', 1))
    config['ddp'] = os.environ.get('RANK', -1) != -1
    config['master_rank'] = 0
    config['is_master'] = config['rank'] == config['master_rank']
    if config['ddp']:
        # set appropriate CUDA device
        torch.cuda.set_device(config['local_rank'])
        # init process group
        dist.init_process_group(backend=config.get('ddp_backend', 'nccl'))  # nccl, gloo, etc

def cleanup_ddp(config: dict[str, Any]):
    if config['ddp']:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
