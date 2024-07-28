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

import torch_xla as xla  # noqa: F401
import torch_xla.amp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as xpl
import torch_xla.distributed.xla_backend  # required for `xla://` init_method and `xla` backend
import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.experimental.pjrt_backend  # required for TPU v2/v3 as TPU v2/v3 support in `torch.distributed` is still experimental
import torch_xla.runtime as xr

import gpt2.opts as opts
import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.meters import XLAAverageMeter
from gpt2.model import GPT, GPTConfig


def train_model(args: argparse.Namespace):
    utils.set_seed(args.seed)
    xm.set_rng_state(args.seed)

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)

    # training device
    device = xm.xla_device()
    device_hw = xm.xla_device_hw(device)

    torch.set_float32_matmul_precision(args.matmul_precision)
    print(f'Set float32 matmul precision to {args.matmul_precision}')

    if args.train_batch_size % xr.world_size() != 0:
        raise ValueError('train_batch_size must be divisible by world_size')
    if args.eval_batch_size % xr.world_size() != 0:
        raise ValueError('eval_batch_size must be divisible by world_size')
    train_batch_size = args.train_batch_size // xr.world_size()
    eval_batch_size = args.eval_batch_size // xr.world_size()
    effective_batch_size = train_batch_size * xr.world_size() * args.gradient_accum_step
    xm.master_print(
        f'Effective batch size: {effective_batch_size} '
        f'(micro_batch_size={train_batch_size}, '
        f'gradient_accum_step={args.gradient_accum_step}, '
        f'num_devices={xr.world_size()})'
    )

    # dataset
    train_lm_dataset = LMDataset(
        args.train_dir,
        train_batch_size,
        args.seq_length,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
    )
    validation_lm_dataset = LMDataset(
        args.valid_dir,
        eval_batch_size,
        args.seq_length,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
    )

    # data loader
    train_data_loader = DataLoader(
        train_lm_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=args.drop_last,
    )
    validation_data_loader = DataLoader(
        validation_lm_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=args.drop_last,
    )

    # device loader
    train_device_loader = xpl.MpDeviceLoader(train_data_loader, device=device)
    validation_device_loader = xpl.MpDeviceLoader(validation_data_loader, device=device)

    # mixed precision training
    # note: AMP only supported for XLA:TPU and XLA:GPU
    mp_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        mp_dtype = torch.bfloat16
    elif isinstance(args.mixed_precision, str):
        raise ValueError(f'Unsupported mixed precision type: {args.mixed_precision}')
    autocast_context = torch_xla.amp.autocast(
        device,
        enabled=((mp_dtype in (torch.float16, torch.bfloat16)) and device_hw != 'CPU'),
        dtype=mp_dtype,
    )
    autocast_enabled = autocast_context._enabled  # pyright: ignore[reportPrivateUsage]
    if not autocast_enabled:
        autocast_context = nullcontext()

    # scaling is not needed for bfoat16
    scaler = torch_xla.amp.GradScaler(enabled=(mp_dtype == torch.float16 and device_hw != 'TPU'))

    # resume from previous checkpoint
    saved_states = None
    if args.from_checkpoint is None:
        gpt_config = GPTConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
            tie_weights=False,
        )
    else:
        xm.master_print(f'Loading states from checkpoint {args.from_checkpoint}')
        # model is saved with xm.save() which moves tensors to CPU before saving,
        # so we can safely discard `map_location`.
        saved_states = torch.load(args.from_checkpoint, map_location=None)
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
        args.tie_weights = args.tie_weights & gpt_config.tie_weights
        gpt_config.tie_weights = False

    model = GPT(gpt_config, device=device)
    model.to(device)
    # tie_weights must be called after moving to device if we are on XLA device,
    # otherwise it will be treated as separate Tensors.
    if args.tie_weights:
        gpt_config.tie_weights = args.tie_weights
        model.use_tied_weights = True
        model.tie_weights()
    criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model,
        device,
        args.optim_type,
        lr=learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay,
        use_syncfree_optim=autocast_enabled and args.use_syncfree_optim,
    )
    if args.decay_method == 'noam':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step, args.d_model, args.warmup_steps,
            ),
        )
    elif args.decay_method == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step, learning_rate, args.min_lr,
                args.warmup_steps,
                args.decay_steps, factor=1/learning_rate,
            ),
        )
    else:
        raise ValueError(f'Unsupported scheduler decay method: {args.decay_method}')

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
    if args.compile:
        xm.master_print('Compiling the model')
        model = torch.compile(model, backend='openxla' if device.type == 'xla' else 'inductor')

    # wrap the model with DDP
    if args.ddp:
        model = DDP(
            model,
            device_ids=[xr.local_ordinal()],
            output_device=xr.local_ordinal(),
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # logging with wandb
    wandb_run = None
    if xm.is_master_ordinal() and args.wandb_logging:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )

    # training loop
    global_step = initial_step
    batch_loss = 0.0
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = XLAAverageMeter('running_losses', device=device)

    xm.master_print(f'Model has {utils.count_model_param(raw_model) / 10 ** 6:0.2f}M parameters')
    train_iter = tqdm(
        range(initial_step, args.train_steps),
        desc=f'{device_hw}:{xr.global_ordinal()} - Training model',
        disable=xr.local_ordinal() != 0,
        ncols=120,
    )

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    while global_step < args.train_steps:
        for batch_idx, (input_ids, labels) in enumerate(train_device_loader):
            if input_ids.dim() == 3:
                assert input_ids.shape[0] == 1
                input_ids = input_ids[0]
            if labels.dim() == 3:
                assert labels.shape[0] == 1
                labels = labels[0]

            if args.ddp:
                # we only sync gradients at the last step of gradient accumulation
                # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
                model.require_backward_grad_sync = (batch_idx + 1) % args.gradient_accum_step == 0

            with autocast_context:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            if args.gradient_accum_step > 1:
                loss /= args.gradient_accum_step
            batch_loss += loss.detach()

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.gradient_accum_step == 0:
                if not args.ddp:
                    xm.reduce_gradients(optimizer)
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # TODO: handle the case when wandb is disabled
                wandb_accum_logs.append({
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
                })
                wandb_accum_logs[-1].update({
                    'loss/batch_loss': batch_loss,
                    'step': global_step,
                })

                lr_scheduler.step()
                running_loss.update(batch_loss)

                if (global_step + 1) % args.valid_interval == 0:
                    xm.rendezvous('all_reduce_running_loss')
                    running_loss.all_reduce()
                    valid_results = eval_model(
                        model,
                        criterion,
                        validation_device_loader,
                        args.valid_steps,
                        autocast_context,
                    )
                    wandb_accum_logs[-1].update({
                        'loss/train': running_loss.average,
                        'loss/valid': valid_results['loss'],
                    })
                    running_loss.reset()

                if (
                    len(wandb_accum_logs) >= args.wandb_logging_interval or
                    (len(wandb_accum_logs) > 0 and global_step + 1 >= args.train_steps)
                ):
                    batch_loss_values = [loss['loss/batch_loss'] for loss in wandb_accum_logs]
                    xm.rendezvous('all_reduce_batch_loss')
                    reduced_batch_loss_values = xm.all_reduce(xm.REDUCE_SUM, torch.tensor(batch_loss_values, device=device), scale=1.0 / xr.world_size())
                    reduced_batch_loss_values = reduced_batch_loss_values.tolist()
                    for idx in range(len(wandb_accum_logs)):
                        wandb_accum_logs[idx]['loss/batch_loss'] = reduced_batch_loss_values[idx]
                    if wandb_run is not None:
                        for log_idx in range(len(wandb_accum_logs)):
                            wandb_run.log(wandb_accum_logs[log_idx])
                    wandb_accum_logs = []
                    xm.rendezvous('exit_wandb_logging')

                if (global_step + 1) % args.save_interval == 0:
                    if xm.is_master_ordinal():
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
                            checkpoints_dir=args.checkpoints_dir,
                            model_basename='gpt2',
                            limit=args.saved_checkpoint_limit,
                        )
                        model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step + 1}.pt')
                        xm.save(checkpoint_dict, model_save_path, master_only=True, global_master=True)
                    xm.rendezvous('save_checkpoint')

                train_iter.set_postfix({
                    'loss': f'{batch_loss:0.3f}',
                })
                batch_loss = 0.0
                global_step += 1
                train_iter.update()
                if global_step >= args.train_steps:
                    break

def _mp_fn(index: int, args: argparse.Namespace) -> None:
    dist.init_process_group(backend='xla', init_method='xla://')
    train_model(args)

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run pre-training GPT2 model with XLA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.add_run_pretrain_xla_opts(parser)
    args = parser.parse_args()

    xmp.spawn(_mp_fn, args=(args,), start_method=args.mp_start_method)

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

    running_loss = XLAAverageMeter('running_loss', device=device)

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
    xm.rendezvous('all_reduce_evaluation_loss')
    running_loss.all_reduce()
    return {
        'loss': running_loss.average,
    }


if __name__ == '__main__':
    main()
