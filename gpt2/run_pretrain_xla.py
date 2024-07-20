"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any

import wandb

import torch
import torch.amp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import torch_xla as xla
import torch_xla.amp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as xpl
import torch_xla.distributed.xla_backend  # required for `xla://` init_method and `xla` backend
import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.experimental.pjrt_backend  # required for TPU v2/v3 as TPU v2/v3 support in `torch.distributed` is still experimental
import torch_xla.runtime as xr

import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.model import GPT, GPTConfig


def train_model(config: dict[str, Any]):
    utils.set_seed(config['seed'])
    xm.set_rng_state(config['seed'])

    checkpoints_dir = utils.ensure_dir(config['checkpoints_dir'])

    # training device
    device = xm.xla_device()
    device_hw = xm.xla_device_hw(device)

    matmul_precision = config.get('matmul_precision', 'highest')
    torch.set_float32_matmul_precision(matmul_precision)
    print(f'Set float32 matmul precision to {matmul_precision}')

    if config['train_batch_size'] % xr.world_size() != 0:
        raise ValueError('train_batch_size must be divisible by world_size')
    if config['eval_batch_size'] % xr.world_size() != 0:
        raise ValueError('eval_batch_size must be divisible by world_size')
    train_batch_size = config['train_batch_size'] // xr.world_size()
    eval_batch_size = config['eval_batch_size'] // xr.world_size()

    # dataset
    train_lm_dataset = LMDataset(
        config['train_dir'],
        train_batch_size,
        config['seq_length'],
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
    )
    validation_lm_dataset = LMDataset(
        config['valid_dir'],
        eval_batch_size,
        config['seq_length'],
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
    )

    # data loader
    train_data_loader = DataLoader(
        train_lm_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=config['drop_last'],
    )
    validation_data_loader = DataLoader(
        validation_lm_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=config['drop_last'],
    )

    # device loader
    train_device_loader = xpl.MpDeviceLoader(train_data_loader, device=device)
    validation_device_loader = xpl.MpDeviceLoader(validation_data_loader, device=device)

    # logging with wandb
    wandb_run = None
    wandb_config = config['wandb']
    if xm.is_master_ordinal() and wandb_config['logging']:
        wandb_run = wandb.init(
            project=wandb_config['project'],
            name=wandb_config['name'],
            config=config,
            tags=wandb_config.get('tags', None),
            notes=wandb_config.get('notes', None),
            id=wandb_config.get('resume_id', None),
            resume='must' if wandb_config.get('resume_id', None) is not None else None,
        )

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
        # TODO: check keys that do not require configuration match
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
            device_ids=[xr.local_ordinal()],
            output_device=xr.local_ordinal(),
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # training loop
    train_steps = config['train_steps']
    valid_steps = config['valid_steps']
    valid_interval = config['valid_interval']
    save_interval = config['save_interval']
    gradient_accum_step = config['gradient_accum_step']

    global_step = initial_step
    batch_loss = 0.0
    batch_losses: list[float | torch.Tensor] = []
    learning_rates: list[list[float]] = []
    running_loss = AverageMeter('running_losses', device=device)
    wandb_logging_interval = config['wandb']['logging_interval']

    xm.master_print(f'Model has {utils.count_model_param(raw_model) / 10 ** 6:0.2f}M parameters')
    train_iter = tqdm(
        range(initial_step, train_steps),
        desc=f'{device_hw}:{xr.global_ordinal()} - Training model',
        disable=xr.local_ordinal() != 0,
        ncols=120,
    )

    # set model in training mode
    model.train()
    while global_step < train_steps:
        for batch_idx, (input_ids, labels) in enumerate(train_device_loader):
            optimizer.zero_grad()

            if input_ids.dim() == 3:
                assert input_ids.shape[0] == 1
                input_ids = input_ids[0]
            if labels.dim() == 3:
                assert labels.shape[0] == 1
                labels = labels[0]

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

            if (batch_idx + 1) % gradient_accum_step == 0:
                if config['max_grad_norm'] > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

                if config['ddp']:
                    scaler.step(optimizer)
                else:
                    if scaler.is_enabled():
                        # reduce gradient manually
                        gradients = xm._fetch_gradients(optimizer)  # pyright: ignore[reportPrivateUsage]
                        xm.all_reduce(xm.REDUCE_SUM, gradients, scale=1.0 / xr.world_size())
                        scaler.step(optimizer)
                    else:
                        xm.optimizer_step(optimizer)

                scaler.update()

                batch_losses.append(batch_loss)
                for group_id, group_lr in enumerate(lr_scheduler.get_last_lr()):
                    if group_id == 0:
                        learning_rates.append([group_lr])
                    else:
                        learning_rates[-1].append(group_lr)

                if (
                    len(batch_losses) % wandb_logging_interval == 0 or
                    (len(batch_losses) > 0 and global_step + 1 >= train_steps)
                ):
                    batch_losses = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(batch_losses, device=device), scale=1.0 / xr.world_size())
                    batch_losses = batch_losses.tolist()
                    if wandb_run is not None:
                        for log_idx in range(len(batch_losses)):
                            log_dict = {}
                            log_dict['loss/batch_loss'] = batch_losses[log_idx]
                            for group_id, group_lr in enumerate(learning_rates[log_idx]):
                                log_dict[f'learning_rate/group_{group_id}'] = group_lr
                            wandb_run.log(log_dict, step=global_step-(len(batch_losses) - log_idx - 1))
                    batch_losses = []
                    learning_rates = []

                lr_scheduler.step()

                running_loss.update(batch_loss)
                train_iter.set_postfix({
                    'loss': f'{batch_loss:0.3f}',
                })
                batch_loss = 0.0

                if (global_step + 1) % valid_interval == 0:
                    running_loss.all_reduce()
                    valid_results = eval_model(
                        model,
                        criterion,
                        validation_device_loader,
                        valid_steps,
                        autocast_context,
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            'loss/train': running_loss.average,
                            'loss/valid': valid_results['loss'],
                        }, step=global_step + 1)
                    running_loss.reset()

                if (global_step + 1) % save_interval == 0 and xm.is_master_ordinal():
                    checkpoint_dict = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': vars(gpt_config),
                        'global_step': global_step + 1,
                    }
                    if scaler.is_enabled():
                        checkpoint_dict['scaler'] = scaler.state_dict()
                    utils.ensure_num_saved_checkpoints(
                        checkpoints_dir=config['checkpoints_dir'],
                        model_basename='gpt2',
                        limit=config['saved_checkpoint_limit'],
                    )
                    model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step + 1}.pt')
                    xm.save(checkpoint_dict, model_save_path, master_only=True, global_master=True)

                global_step += 1
                train_iter.update()
                if global_step >= train_steps:
                    break

def _mp_fn(index: int, config: dict[str, Any]) -> None:
    dist.init_process_group(backend='xla', init_method='xla://')
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

    xmp.spawn(_mp_fn, args=(config,), start_method=config['mp_start_method'])

@torch.no_grad()
def eval_model(
    model,
    criterion,
    eval_data_loader,
    valid_steps: int,
    autocast_context=nullcontext,
) -> dict[str, float]:
    device = model.device
    device_hw = xm.xla_device_hw(device)
    progess_bar = tqdm(
        range(valid_steps),
        desc=f'{device_hw}:{xr.global_ordinal()} - Evaluating model',
        disable=xr.local_ordinal() != 0,
        ncols=120,
    )

    running_loss = AverageMeter('running_loss', device=device)

    # set model in evaluation mode
    is_training = model.training
    model.eval()
    for batch_idx, (input_ids, labels) in enumerate(eval_data_loader):
        if input_ids.dim() == 3:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if labels.dim() == 3:
            assert labels.shape[0] == 1
            labels = labels[0]
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        running_loss.update(loss.detach())
        progess_bar.set_postfix({'loss': f'{loss:0.3f}'})
        progess_bar.update()
        if (batch_idx + 1) >= valid_steps:
            break

    # set model back to the original mode
    model.train(is_training)

    running_loss.all_reduce()
    return {
        'loss': running_loss.average,
    }

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
            coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)]
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
