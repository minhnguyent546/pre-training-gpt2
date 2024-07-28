"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
import time
from contextlib import nullcontext
from tqdm.autonotebook import tqdm
from typing import Any

import wandb

import torch
import torch.amp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpt2.opts as opts
import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.meters import AverageMeter
from gpt2.model import GPT, GPTConfig


def train_model(args: argparse.Namespace) -> None:
    utils.set_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if args.is_master:
        print(f'Set float32 matmul precision to {args.matmul_precision}')

    checkpoints_dir = utils.ensure_dir(args.checkpoints_dir)

    if args.train_batch_size % args.world_size != 0:
        raise ValueError('train_batch_size must be divisible by world_size')
    if args.eval_batch_size % args.world_size != 0:
        raise ValueError('eval_batch_size must be divisible by world_size')
    train_batch_size = args.train_batch_size // args.world_size
    eval_batch_size = args.eval_batch_size // args.world_size
    effective_batch_size = train_batch_size * args.world_size * args.gradient_accum_step
    if args.is_master:
        print(
            f'Effective batch size: {effective_batch_size} '
            f'(micro_batch_size={train_batch_size}, '
            f'gradient_accum_step={args.gradient_accum_step}, '
            f'num_devices={args.world_size})'
        )

    # dataset
    train_dataset = LMDataset(
        args.train_dir,
        train_batch_size,
        args.seq_length,
        num_replicas=args.world_size,
        rank=args.rank,
    )
    validation_dataset = LMDataset(
        args.valid_dir,
        eval_batch_size,
        args.seq_length,
        num_replicas=args.world_size,
        rank=args.rank,
    )

    # logging with wandb
    wandb_run = None
    if args.is_master and args.wandb_logging:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            id=args.wandb_resume_id,
            resume='must' if args.wandb_resume_id is not None else None,
        )

    # training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == 'cuda' and args.mixed_precision == 'fp16':
        mp_dtype = torch.float16
        if args.is_master:
            print('Mixed precision training is enabled with fp16')
    elif device.type == 'cuda' and args.mixed_precision == 'bf16':
        if torch.cuda.is_bf16_supported():
            mp_dtype = torch.bfloat16
            if args.is_master:
                print('Mixed precision training is enabled with bf16')
        else:
            mp_dtype = torch.float16
            if args.is_master:
                print('bf16 is not supported on your hardware, fallback to fp16')
    autocast_context = torch.cuda.amp.autocast(enabled=(mp_dtype in (torch.float16, torch.bfloat16)), dtype=mp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(mp_dtype == torch.float16))

    # resume from previous checkpoint
    pretrained_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
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
            tie_weights=args.tie_weights,
        )
        model = GPT(gpt_config)
    elif args.from_checkpoint in pretrained_models:
        if args.is_master:
            print(f'Loading states from pretrained model {args.from_checkpoint}')
        gpt_config = GPTConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
            tie_weights=args.tie_weights,
        )
        if args.ddp_enabled:
            if args.local_rank == 0:
                # make sure the checkpoint is downloaded only once by the local master process
                model = GPT.from_pretrained(args.from_checkpoint, gpt_config)
                dist.barrier()
            else:
                dist.barrier()
                model = GPT.from_pretrained(args.from_checkpoint, gpt_config)
        else:
            model = GPT.from_pretrained(args.from_checkpoint, gpt_config)
        model.truncate_seq_length(args.seq_length)
        gpt_config.seq_length = args.seq_length
    else:
        if args.is_master:
            print(f'Loading states from checkpoint {args.from_checkpoint}')
        saved_states = torch.load(args.from_checkpoint, map_location=device)
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
    if model.config.tie_weights:
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
        model.load_state_dict(saved_states['model'])
        optimizer.load_state_dict(saved_states['optimizer'])
        lr_scheduler.load_state_dict(saved_states['lr_scheduler'])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states['scaler'])
        if 'global_step' in saved_states:
            initial_step = saved_states['global_step']

    raw_model = model
    # compile the model
    if args.compile:
        if args.is_master:
            print('Compiling the model')
        model = torch.compile(model)

    # convert the model to distributed data parallel
    if args.ddp_enabled:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.do_test:
        valid_results = eval_model(
            model,
            device,
            criterion,
            validation_dataset,
            args.valid_steps,
            args,
            autocast_context,
        )
        if args.is_master:
            print('** Testing results **')
            print(f'Loss: {valid_results["loss"]}')
            print(f'Perplexity: {utils.get_perplexity(valid_results["loss"])}')
        return

    # training loop
    num_tokens_per_batch = train_batch_size * args.gradient_accum_step * args.seq_length

    if args.is_master:
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'Model has {num_parameters / 10 ** 6:0.2f}M parameters')

    if args.ddp_enabled:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc=f'GPU{args.rank} - Training model',
            disable=args.local_rank != 0,
            ncols=120,
        )
    else:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc='Training model',
            ncols=120,
        )

    global_step = initial_step
    batch_loss = 0.0
    batch_fb_time = 0.0  # batch forward + backward time
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter('running_loss', device=device)

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    while global_step < args.train_steps:
        for batch_idx, (input_ids, labels) in enumerate(train_dataset):
            ts = time.perf_counter()
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if args.ddp_enabled:
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

            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_fb_time += time.perf_counter() - ts

            if (batch_idx + 1) % args.gradient_accum_step == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                batch_throughput = num_tokens_per_batch / batch_fb_time
                batch_throughput *= args.world_size  # estimate throughput across devices

                # TODO: handle the case when wandb is disabled
                wandb_accum_logs.append({
                    f'learning_rate/group_{group_id}': group_lr
                    for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
                })
                wandb_accum_logs[-1].update({
                    'loss/batch_loss': batch_loss,
                    'throughput': batch_throughput,
                    'step': global_step,
                })

                lr_scheduler.step()
                running_loss.update(batch_loss)

                if (global_step + 1) % args.valid_interval == 0:
                    if args.ddp_enabled:
                        running_loss.reduce(dst=args.master_rank)
                    valid_results = eval_model(
                        model,
                        device,
                        criterion,
                        validation_dataset,
                        args.valid_steps,
                        args,
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
                    batch_loss_values = torch.tensor(
                        [loss['loss/batch_loss'] for loss in wandb_accum_logs],
                        dtype=torch.float32,
                        device=device,
                    )
                    dist.all_reduce(batch_loss_values, op=dist.ReduceOp.AVG)
                    reduced_batch_loss_values = batch_loss_values.tolist()
                    for idx in range(len(wandb_accum_logs)):
                        wandb_accum_logs[idx]['loss/batch_loss'] = reduced_batch_loss_values[idx]
                    if wandb_run is not None:
                        for log_idx in range(len(wandb_accum_logs)):
                            wandb_run.log(wandb_accum_logs[log_idx])
                    wandb_accum_logs = []
                    dist.barrier()

                if (global_step + 1) % args.save_interval == 0:
                    if args.is_master:
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
                            args.checkpoints_dir,
                            'gpt2',
                            args.saved_checkpoint_limit,
                        )
                        model_save_path = os.path.join(checkpoints_dir, f'gpt2-{global_step + 1}.pt')
                        torch.save(checkpoint_dict, model_save_path)
                    dist.barrier()

                train_iter.set_postfix({
                    'loss': f'{batch_loss:0.3f}',
                    'throughput': f'{batch_throughput:0.3f} tokens/s'
                })
                batch_loss = 0.0
                batch_fb_time = 0.0
                global_step += 1
                train_iter.update()
                if global_step >= args.train_steps:
                    break

def main():
    parser = argparse.ArgumentParser(
        description='Run pre-training GPT2 model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.add_run_pretrain_opts(parser)
    args = parser.parse_args()

    setup_ddp(args)

    train_model(args)

    cleanup_ddp(args)

@torch.no_grad()
def eval_model(
    model: GPT | DDP,
    device: torch.device,
    criterion,
    eval_dataset: LMDataset,
    valid_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
) -> dict[str, float]:
    evaluation_loss = AverageMeter('evaluation_loss', device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    if args.ddp_enabled:
        progress_bar = tqdm(
            range(valid_steps),
            total=valid_steps,
            desc=f'GPU{args.rank} - Evaluating model',
            disable=args.local_rank != 0,
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

    if args.ddp_enabled:
        evaluation_loss.reduce(dst=args.master_rank)

    return {
        'loss': evaluation_loss.average,
    }

def setup_ddp(args: argparse.Namespace) -> None:
    args.rank = int(os.environ.get('RANK', 0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.ddp_enabled = os.environ.get('RANK', -1) != -1
    args.master_rank = 0
    args.is_master = args.rank == args.master_rank
    if args.ddp_enabled:
        # set appropriate CUDA device
        torch.cuda.set_device(args.local_rank)
        # init process group
        dist.init_process_group(backend=getattr(args, 'ddp_backend', 'nccl'))  # nccl, gloo, etc

def cleanup_ddp(args: argparse.Namespace) -> None:
    if args.ddp_enabled:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
