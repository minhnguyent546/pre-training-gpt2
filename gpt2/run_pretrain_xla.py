"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
import time
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any, Literal, Iterator

import wandb

import torch
import torch.amp
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_xla as xla
import torch_xla.amp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as xpl
import torch_xla.distributed.xla_backend  # required for `xla://` init_method and `xla` backend
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend  # required for TPU v2/v3 as TPU v2/v3 support in `torch.distributed` is still experimental
import torch_xla.runtime as xr

import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.model import GPT, GPTConfig


def train_model(config: dict[str, Any]):
    utils.set_seed(config['seed'])
    matmul_precision = config.get('matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    xm.master_print(f'Set float32 matmul precision to {matmul_precision}')

    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # dataset
    train_dataset = LMDataset(
        config['train_dir'],
        config['train_batch_size'],
        config['seq_length'],
        shuffle=False,
        num_replicas=config['world_size'],
        rank=config['rank'],
    )
    validation_dataset = LMDataset(
        config['valid_dir'],
        config['eval_batch_size'],
        config['seq_length'],
        shuffle=False,
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
    device = xm.xla_device()
    device_hw = xm.xla_device_hw(device)

    # mixed precision training
    # note: AMP only supported for XLA:TPU and XLA:GPU
    mp_dtype = torch.float32
    if config['mixed_precision'] == 'fp16':
        mp_dtype = torch.float16
    elif config['mixed_precision'] == 'bf16':
        mp_dtype = torch.bfloat16
    elif isinstance(config['mixed_precision'], str):
        raise ValueError(f'Unsupported mixed precision type: {config["mixed_precision"]}')
    autocast_context = torch_xla.amp.autocast(
        device,
        enabled=(mp_dtype in (torch.float16, torch.bfloat16)),
        dtype=mp_dtype,
    )
    autocast_enabled = autocast_context._enabled  # pyright: ignore[reportPrivateUsage]
    if not autocast_enabled:
        autocast_context = nullcontext()
    # scaling is not needed for bfoat16
    scaler = torch_xla.amp.GradScaler(enabled=(mp_dtype == torch.float16 and device_hw != 'TPU'))
    if scaler.is_enabled():
        xm.master_print('Gradient scaler is enabled')
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
        xm.master_print(f'Loading states from checkpoint {preload_checkpoint}')
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

    model = GPT(gpt_config, device=device)
    model.to(device)
    # tie_weights must be called after moving to device if we are on XLA device,
    # otherwise it will be treated as separate Tensors.
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
        use_syncfree_optim=autocast_enabled and config['use_syncfree_optim'],
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
        unwanted_prefixes = ['module.', '_orig_mod.']  # created by DDP() and torch.compile()
        for prefix in unwanted_prefixes:
            consume_prefix_in_state_dict_if_present(saved_states['model'], prefix=prefix)
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optimizer'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states['scaler'])
        if 'global_step' in saved_states:
            initial_step = saved_states['global_step']
        if 'running_losses' in saved_states:
            running_losses = saved_states['running_losses']
            if config['rank'] > len(running_losses) - 1:
                raise RuntimeError('rank is out of running_losses range founded in checkpoint')
            running_loss = AverageMeter(**running_losses[config['rank']])
            running_loss.device = device

    # initialization is nondeterministic with multiple threads in PjRt.
    # synchronize model parameters across replicas manually.
    # optional for TPUv4 and GPU
    xm.broadcast_master_param(model)

    raw_model = model
    # compile the model
    if config['compile']:
        xm.master_print('Compiling the model')
        model = torch.compile(model, backend='openxla' if device.type == 'xla' else 'inductor')

    # wrap the model with DDP
    if config['ddp']:
        model = DDP(
            model,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # training loop
    train_steps = config['train_steps']
    valid_steps = config['valid_steps']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    gradient_accum_step = config['gradient_accum_step']
    num_tokens_per_batch = config['train_batch_size'] * gradient_accum_step * config['seq_length']

    xm.master_print(f'Model has {utils.count_model_param(raw_model) / 10 ** 6:0.2f}M parameters')

    train_iter = tqdm(
        range(initial_step, train_steps),
        desc=f'{device_hw}:{config["rank"]} - Training model',
        disable=config['local_rank'] != 0,
        ncols=120,
    )

    batch_loss = 0.0
    batch_idx = 0
    batch_fb_time = 0.0  # batch forward + backward time
    global_step = initial_step
    model.train()
    while global_step < train_steps:
        optimizer.zero_grad()

        ts = time.monotonic()
        input_ids, labels = train_dataset.next_batch()
        input_ids = input_ids.type(torch.int64).to(device)
        labels = labels.type(torch.int64).to(device)

        if config['ddp']:
            # we only sync gradients at the last step of gradient accumulation
            # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
            model.require_backward_grad_sync = (batch_idx + 1) % gradient_accum_step == 0
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        if gradient_accum_step > 1:
            loss /= gradient_accum_step
        # batch_loss += loss
        batch_loss += loss.detach()
        # batch_loss += loss.item()

        scaler.scale(loss).backward()
        utils.sync_host_device(device)
        batch_fb_time += time.monotonic() - ts

        if (batch_idx + 1) % gradient_accum_step == 0:
            if config['max_grad_norm'] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

            if config['ddp']:
                scaler.step(optimizer)
                xm.mark_step()
            else:
                if scaler.is_enabled():
                    # reduce gradient manually
                    gradients = xm._fetch_gradients(optimizer)  # pyright: ignore[reportPrivateUsage]
                    xm.all_reduce(xm.REDUCE_SUM, gradients, scale=1.0 / config['world_size'])
                    scaler.step(optimizer)
                else:
                    xm.optimizer_step(optimizer, barrier=True)
            scaler.update()

            batch_throughput = num_tokens_per_batch / batch_fb_time
            batch_throughput *= config['world_size'] # estimate throughput when training with multiple gpus

            if wandb_run is not None:
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                    wandb_run.log({f'learning_rate/group-{group_id}': group_lr}, step=global_step)
                wandb_run.log({
                    'loss/batch_loss': batch_loss,
                    'throughput': batch_throughput,
                }, step=global_step)

            lr_scheduler.step()

            # running_loss.update(batch_loss)
            train_iter.set_postfix({
                'loss': f'{batch_loss:0.3f}',
                'throughput': f'{batch_throughput:0.3f} tokens/s'
            })
            batch_loss = 0.0
            batch_fb_time = 0.0

            if (global_step + 1) % valid_interval == 0:
                running_loss.all_reduce()
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
                running_losses = running_loss.xla_all_gather_object(config['world_size'])
                if config['is_master']:
                    checkpoint_dict = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': vars(gpt_config),
                        'global_step': global_step + 1,
                        'running_losses': running_losses,
                    }
                    if scaler.is_enabled():
                        checkpoint_dict['scaler'] = scaler.state_dict()
                    utils.ensure_num_saved_checkpoints(
                        config['checkpoints_dir'],
                        'gpt2',
                        config['saved_checkpoint_limit'],
                    )
                    model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step + 1}.pt')
                    xm.save(checkpoint_dict, model_save_path, master_only=True, global_master=True)

            global_step += 1
            train_iter.update()

        batch_idx += 1

def _mp_fn(index: int, config: dict[str, Any]) -> None:
    setup_pjrt(config)

    train_model(config)

def main() -> None:
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

    nprocs = 1 if config['xla_one_core'] else None  # pjrt only support 1 or all cores.
    xmp.spawn(_mp_fn, args=(config,), nprocs=nprocs, start_method=config['mp_start_method'])

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
    accum_loss = 0.0
    for _ in valid_iter:
        input_ids, labels = eval_dataset.next_batch()
        input_ids = input_ids.type(torch.int64).to(device)
        labels = labels.type(torch.int64).to(device)
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        accum_loss += loss.detach()
        valid_iter.set_postfix({'loss': f'{loss:0.3f}'})

    print('Done evaluation')
    accum_loss = accum_loss.item()
    print(f"{accum_loss = }")
    exit()
    evaluation_loss = AverageMeter(sum=accum_loss, count=len(valid_iter), device=device)
    print(f"{evaluation_loss.average = }")

    print('Start reducing')
    evaluation_loss.all_reduce()
    print('End reducing')
    model.train(is_training)

    return {
        'loss': evaluation_loss.average,
    }

def setup_pjrt(config: dict[str, Any]) -> None:
    dist.init_process_group(backend='xla', init_method='xla://')
    config['rank'] = xm.get_ordinal()
    config['master_rank'] = 0
    config['is_master'] = xm.is_master_ordinal()
    config['local_rank'] = xm.get_local_ordinal()
    config['world_size'] = xm.xrt_world_size()

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
        name: str = '',
        sum: int | float = 0.0,
        count: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.name = name
        self.sum = sum
        self.count = count
        self.device = device

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def update(self, value: int | float, nums: int = 1) -> None:
        self.sum += value * nums
        self.count += nums

    def reduce(self, dst: int) -> None:
        """
        Perform in-place reduce

        Work on both CUDA and XLA device. On XLA device, it will call to
        `ProcessGroup.reduce`, which will call to `reduce` in torch_xla/distributed/xla_backend.py module
        (https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_backend.py#L167).

        Currently, which torch_xla version 2.3.1, this function raises `NotImplementedError`
        on XLA device, prefer to use `all_reduce` instead.

        """
        torch_xla.distributed.xla_backend
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        """
        Perform in-place all-reduce

        Work on both CUDA and XLA device. On XLA device, it will call to
        `ProcessGroup.all_reduce`, which will call to `allreduce` in torch_xla/distributed/xla_backend.py module
        (https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_backend.py#L67).

        """
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def gather_object(self, dst: int, world_size: int, is_master: bool) -> list[dict[str, Any]] | None:
        if self._is_xla_device():
            raise NotImplementedError('`gather_object` is not supported on XLA device.')
        output = [None for _ in range(world_size)] if is_master else None
        object_dict = self.to_dict()
        dist.gather_object(object_dict, output, dst)
        assert output is not None if is_master else None
        return output

    def all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        if self._is_xla_device():
            raise NotImplementedError(
                '`all_gather_object` is not supported on XLA device. ',
                'Please use `xla_all_gather_object` instead.'
            )
        output = [None for _ in range(world_size)]
        object_dict = self.to_dict()
        dist.all_gather_object(output, object_dict)
        return output

    def xla_all_gather_object(self, world_size: int) -> list[dict[str, Any]]:
        """
        Modified from `torch/distributed/distributed_c10d.py`, work on both CUDA and XLA device.

        Note this function is experimental, use with caution.
        """
        input_tensor, local_size = utils.object_to_tensor(self.to_dict(), self.device)

        object_sizes_tensor = torch.zeros(
            world_size, dtype=torch.long, device=self.device,
        )
        object_size_list = [
            object_sizes_tensor[i].unsqueeze(dim=0) for i in range(world_size)
        ]
        dist.all_gather(object_size_list, local_size)
        max_object_size = int(max(object_size_list).item())  # pyright: ignore

        # resize tensor to max size across all ranks.
        input_tensor.resize_(max_object_size)
        coalesced_output_tensor = torch.empty(
            max_object_size * world_size, dtype=torch.uint8, device=self.device,
        )
        # output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
            for i in range(world_size)
        ]
        dist.all_gather(output_tensors, input_tensor)

        # deserialize outputs back to object.
        object_list = [None for _ in range(world_size)]
        for i, tensor in enumerate(output_tensors):
            tensor = tensor.type(torch.uint8)
            tensor_size = object_size_list[i]
            object_list[i] = utils.tensor_to_object(tensor, tensor_size)
        return object_list

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:
        str_repr = f'{self.__class__.__name__}('
        if self.name:
            str_repr += f'name={self.name}, '
        str_repr += (
            f'value={self.value}, '
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}, '
            f'device={self.device})'
        )
        return str_repr

    def to_dict(self) -> dict[str, Any]:
        return vars(self)

    def _is_xla_device(self) -> bool:
        return self.device is not None and self.device.type == 'xla'


if __name__ == '__main__':
    main()
