"""Pretraining GPT-2 with language modeling objective."""

import argparse
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.version
import wandb
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.autonotebook import tqdm

import gpt2.opts as opts
import gpt2.utils as utils
from gpt2.lm_dataset import LMDataset
from gpt2.meters import AverageMeter
from gpt2.model import GPT, GPTConfig


def train_model(args: argparse.Namespace) -> None:
    # set seed
    utils.set_seed(args.seed)

    # checkpoint dir and log file
    checkpoints_dir_basename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.wandb_logging and args.wandb_name is not None:
        checkpoints_dir_basename += f"-{args.wandb_name}"

    checkpoints_dir = os.path.join(
        args.checkpoints_dir,
        checkpoints_dir_basename,
    )
    os.makedirs(checkpoints_dir, exist_ok=True)
    if args.wandb_logging and args.wandb_name is not None:
        log_file = os.path.join(checkpoints_dir, f"{args.wandb_name}.log")
    else:
        log_file = os.path.join(checkpoints_dir, "training.log")

    def master_print(message: str, console: bool = True) -> None:
        if args.is_master:
            if console:
                print(message)
            with open(log_file, "a") as f:
                print(message, file=f)

    master_print(f"Python version: {sys.version}")
    master_print(
        f"Pytorch version {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
    )
    master_print(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ).stdout
    )
    master_print(f"Args: {vars(args)}")

    # training device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    master_print(f"Using device: {device}")

    torch.set_float32_matmul_precision(args.matmul_precision)
    master_print(f"Set float32 matmul precision to {args.matmul_precision}")

    if args.train_batch_size % args.world_size != 0:
        raise ValueError("train_batch_size must be divisible by world_size")
    if args.eval_batch_size % args.world_size != 0:
        raise ValueError("eval_batch_size must be divisible by world_size")
    train_batch_size = args.train_batch_size // args.world_size
    eval_batch_size = args.eval_batch_size // args.world_size
    effective_batch_size = train_batch_size * args.world_size * args.gradient_accum_step
    tokens_per_fwdbwd = (
        train_batch_size * args.world_size * args.seq_length * args.gradient_accum_step
    )
    master_print(
        f"Effective batch size: {effective_batch_size} "
        f"(micro_batch_size={train_batch_size}, "
        f"gradient_accum_step={args.gradient_accum_step}, "
        f"num_devices={args.world_size})"
    )
    master_print(f"Tokens per forward/backward pass: {tokens_per_fwdbwd}")

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
            resume="must" if args.wandb_resume_id is not None else None,
        )
        master_print(
            f"Wandb logging enabled. project: {args.wandb_project}, name: {args.wandb_name}, id: {wandb_run.id}"
        )

    # mixed precision training
    mp_dtype = torch.float32
    if device.type == "cuda":
        if args.mixed_precision == "fp16":
            mp_dtype = torch.float16
            master_print("Mixed precision training is enabled with fp16")
        elif args.mixed_precision == "bf16":
            if torch.cuda.is_bf16_supported():
                mp_dtype = torch.bfloat16
                master_print("Mixed precision training is enabled with bf16")
            else:
                mp_dtype = torch.float16
                master_print("bf16 is not supported on your hardware, fallback to fp16")
    autocast_context = torch.amp.autocast_mode.autocast(
        device_type=device.type,
        dtype=mp_dtype,
        enabled=(mp_dtype in (torch.float16, torch.bfloat16)),
    )
    autocast_enabled = autocast_context._enabled  # pyright: ignore[reportPrivateUsage]
    if not autocast_enabled:
        autocast_context = nullcontext()

    scaler = torch.amp.grad_scaler.GradScaler(
        device=device.type, enabled=(mp_dtype == torch.float16)
    )

    # resume from previous checkpoint
    pretrained_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
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
            tie_weights=args.tie_weights,
            attn_logit_softcapping=args.attn_logit_softcapping,
            final_logit_softcapping=args.final_logit_softcapping,
        )
        model = GPT(gpt_config)
    elif args.from_checkpoint in pretrained_models:
        master_print(f"Loading states from pretrained model {args.from_checkpoint}")
        gpt_config = GPTConfig(
            vocab_size=args.vocab_size,
            seq_length=args.seq_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            tie_weights=args.tie_weights,
            attn_logit_softcapping=args.attn_logit_softcapping,
            final_logit_softcapping=args.final_logit_softcapping,
        )
        if args.ddp_enabled:
            if args.is_local_master:
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
        master_print(f"Loading states from checkpoint {args.from_checkpoint}")
        saved_states = torch.load(args.from_checkpoint, map_location=device)
        required_keys = ["model", "optimizer", "lr_scheduler", "config"]
        if scaler.is_enabled():
            required_keys.append("scaler")
        for key in required_keys:
            if key not in saved_states:
                raise ValueError(f'Missing key "{key}" in checkpoint')
        gpt_config = GPTConfig(**saved_states["config"])
        model = GPT(gpt_config)

    model.to(device)
    if model.config.tie_weights:
        model.tie_weights()

    if os.getenv("USE_FLASH_ATTN") == "1":
        master_print("USE_FLASH_ATTN is set, trying to use flash attention if available")
    master_print(model)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    eval_criterion = nn.CrossEntropyLoss()
    learning_rate = args.learning_rate
    optimizer = utils.make_optimizer(
        model=model,
        device=device,
        optim_type=args.optim_type,
        lr=learning_rate,
        betas=args.adam_betas,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
    )
    if args.lr_schedule == "noam":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.noam_decay(
                step,
                args.d_model,
                args.warmup_steps,
            ),
        )
    elif args.lr_schedule == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: utils.cosine_decay(
                step,
                learning_rate,
                args.min_lr,
                args.warmup_steps,
                args.decay_steps,
                factor=1 / learning_rate,
            ),
        )
    elif args.lr_schedule == "wsd":
        lr_scheduler = utils.get_wsd_schedule(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_stable_steps=args.stable_steps,
            num_decay_steps=args.decay_steps,
            min_lr_ratio=args.min_lr / learning_rate,
            decay_type=args.decay_type,
        )
    else:
        raise ValueError(f"Unsupported scheduler decay method: {args.lr_schedule}")

    initial_step = 0
    if saved_states is not None:
        unwanted_prefixes = ["module.", "_orig_mod."]  # created by DDP() and torch.compile()
        for prefix in unwanted_prefixes:
            consume_prefix_in_state_dict_if_present(saved_states["model"], prefix=prefix)

        model.load_state_dict(saved_states["model"])
        optimizer.load_state_dict(saved_states["optimizer"])
        lr_scheduler.load_state_dict(saved_states["lr_scheduler"])
        if scaler.is_enabled():
            scaler.load_state_dict(saved_states["scaler"])
        if "global_step" in saved_states:
            initial_step = saved_states["global_step"]

    raw_model = model
    # compile the model
    if args.compile:
        master_print("Compiling the model")
        model = torch.compile(model, dynamic=False, fullgraph=True)

    # wrap the model with DDP
    if args.ddp_enabled:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.do_test:
        valid_results = eval_model(
            model=model,
            device=device,
            criterion=eval_criterion,
            eval_dataset=validation_dataset,
            eval_steps=args.valid_steps,
            args=args,
            autocast_context=autocast_context,
        )
        master_print("** Testing results **")
        master_print(f"Loss: {valid_results['loss']}")
        master_print(f"Perplexity: {utils.get_perplexity(valid_results['loss'])}")
        return

    if args.is_master:
        num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
        master_print(f"Model has {num_parameters / 10**6:0.2f}M parameters")

    if args.ddp_enabled:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc=f"GPU{args.rank}-Training",
            disable=(not args.is_local_master),
            ncols=120,
        )
    else:
        train_iter = tqdm(
            range(initial_step, args.train_steps),
            desc="Training",
            ncols=120,
        )

    # training loop
    global_step = initial_step
    wandb_accum_logs: list[dict[str, Any]] = []
    running_loss = AverageMeter("running_loss", device=device)
    token_seen: int = 0

    # set model in training mode
    model.train()
    optimizer.zero_grad()
    train_loader_iter = iter(train_dataset)
    while global_step < args.train_steps:
        num_items_in_batch = torch.tensor(0, device=device)
        batch_fb_time = 0.0  # batch forward + backward time
        batch_loss = 0.0
        ts = time.perf_counter()
        for batch_idx in range(args.gradient_accum_step):
            try:
                input_ids, labels = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_dataset)
                input_ids, labels = next(train_loader_iter)

            if input_ids.dim() == 3:
                assert input_ids.shape[0] == 1
                input_ids = input_ids[0]
            if labels.dim() == 3:
                assert labels.shape[0] == 1
                labels = labels[0]

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # TODO: assume padding token id is -100, replace with actual padding token id if different
            num_items_in_batch += (labels != -100).sum()

            if args.ddp_enabled:
                # we only sync gradients at the last step of gradient accumulation
                # we can use the below trick or model.no_sync context manager (see: https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1404)
                model.require_backward_grad_sync = batch_idx + 1 == args.gradient_accum_step

            with autocast_context:
                logits = model(input_ids)
                loss = criterion(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))

            scaler.scale(loss).backward()
            batch_loss += loss.detach()

        batch_loss = batch_loss / num_items_in_batch

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_fb_time += time.perf_counter() - ts
        batch_throughput = tokens_per_fwdbwd / batch_fb_time
        batch_throughput *= args.world_size  # estimate throughput across devices
        token_seen += tokens_per_fwdbwd

        scaler.unscale_(optimizer)
        grad_norm_value = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=(args.max_grad_norm if args.max_grad_norm > 0 else float("inf")),
            norm_type=2,
        )
        if bool(torch.isinf(grad_norm_value)) or bool(torch.isnan(grad_norm_value)):
            grad_norm_value = -1

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # TODO: handle the case when wandb is disabled
        wandb_accum_logs.append({
            f"learning_rate/group_{group_id}": group_lr
            for group_id, group_lr in enumerate(lr_scheduler.get_last_lr())
        })
        wandb_accum_logs[-1].update({
            "loss/batch_loss": batch_loss,
            "grad_norm": grad_norm_value,
            "throughput": batch_throughput,
            "token_seen": token_seen,
            "step": global_step,
        })

        lr_scheduler.step()
        running_loss.update(batch_loss, num_items_in_batch)  # pyright: ignore[reportArgumentType]

        # run validation
        if (global_step + 1) % args.valid_interval == 0:
            if args.ddp_enabled:
                running_loss.reduce(dst=args.master_rank)
            valid_results = eval_model(
                model=model,
                device=device,
                criterion=eval_criterion,
                eval_dataset=validation_dataset,
                eval_steps=args.valid_steps,
                args=args,
                autocast_context=autocast_context,
            )
            wandb_accum_logs[-1].update({
                "loss/train": running_loss.average,
                "loss/valid": valid_results["loss"],
            })
            master_print(
                f"[step {global_step + 1} / {args.train_steps}] running_loss: {running_loss.average:0.4f} | valid loss: {valid_results['loss']:0.4f}"
            )
            running_loss.reset()

        # log to wandb
        if len(wandb_accum_logs) >= args.wandb_logging_interval or (
            len(wandb_accum_logs) > 0 and global_step + 1 >= args.train_steps
        ):
            if args.ddp_enabled:
                batch_loss_values = torch.tensor(
                    [loss["loss/batch_loss"] for loss in wandb_accum_logs],
                    dtype=torch.float32,
                    device=device,
                )
                dist.all_reduce(batch_loss_values, op=dist.ReduceOp.AVG)
                reduced_batch_loss_values = batch_loss_values.tolist()
                for idx in range(len(wandb_accum_logs)):
                    wandb_accum_logs[idx]["loss/batch_loss"] = reduced_batch_loss_values[idx]
            if wandb_run is not None:
                for log_idx in range(len(wandb_accum_logs)):
                    wandb_run.log(wandb_accum_logs[log_idx])
            wandb_accum_logs = []

            if args.ddp_enabled:
                dist.barrier()

        # save checkpoint
        if (global_step + 1) % args.save_interval == 0:
            if args.is_master:
                checkpoint_dict = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": vars(gpt_config),
                    "global_step": global_step + 1,
                }
                if scaler.is_enabled():
                    checkpoint_dict["scaler"] = scaler.state_dict()
                utils.ensure_num_saved_checkpoints(
                    checkpoints_dir=args.checkpoints_dir,
                    model_basename="gpt2",
                    limit=args.saved_checkpoint_limit,
                )
                model_save_path = os.path.join(checkpoints_dir, f"gpt2-{global_step + 1}.pt")
                torch.save(checkpoint_dict, model_save_path)

            if args.ddp_enabled:
                dist.barrier()

        if (global_step + 1) % args.log_interval == 0 or global_step + 1 == args.train_steps:
            master_print(
                f"[step {global_step + 1} / {args.train_steps}] loss: {batch_loss:0.4f} | throughput: {batch_throughput:0.1f} tokens/s | grad_norm: {grad_norm_value:0.4f} | token_seen: {token_seen:0.2e}",
                console=False,
            )

        train_iter.set_postfix({
            "loss": f"{batch_loss:0.3f}",
            "grad_norm": f"{grad_norm_value:0.3f}",
            "token_seen": f"{token_seen:0.2e}",
        })
        global_step += 1
        train_iter.update()

        # also save the model at the last step
        if global_step == args.train_steps and args.train_steps % args.save_interval != 0:
            if args.is_master:
                checkpoint_dict = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "config": vars(gpt_config),
                    "global_step": global_step + 1,
                }
                if scaler.is_enabled():
                    checkpoint_dict["scaler"] = scaler.state_dict()
                utils.ensure_num_saved_checkpoints(
                    checkpoints_dir=args.checkpoints_dir,
                    model_basename="gpt2",
                    limit=args.saved_checkpoint_limit,
                )
                model_save_path = os.path.join(checkpoints_dir, f"gpt2-{global_step + 1}.pt")
                torch.save(checkpoint_dict, model_save_path)

            if args.ddp_enabled:
                dist.barrier()


def main():
    parser = argparse.ArgumentParser(
        description="Run pre-training GPT2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    eval_dataset,
    eval_steps: int,
    args: argparse.Namespace,
    autocast_context=None,
    show_progress_bar: bool = False,
) -> dict[str, float]:
    evaluation_loss = AverageMeter("evaluation_loss", device=device)
    if autocast_context is None:
        autocast_context = nullcontext()

    if args.ddp_enabled:
        progress_bar = tqdm(
            total=eval_steps,
            desc=f"GPU{args.rank}-Eval",
            disable=(not args.is_local_master) or (not show_progress_bar),
            position=1,
            leave=False,
            ncols=120,
        )
    else:
        progress_bar = tqdm(
            total=eval_steps,
            desc="Evaluating",
            disabled=(not show_progress_bar),
            position=1,
            leave=False,
            ncols=120,
        )

    # set model in evaluation mode
    is_training = model.training
    model.eval()

    for batch_idx, (input_ids, labels) in enumerate(eval_dataset):
        if input_ids.dim() == 3:
            assert input_ids.shape[0] == 1
            input_ids = input_ids[0]
        if labels.dim() == 3:
            assert labels.shape[0] == 1
            labels = labels[0]

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        num_items_in_batch = (labels != -100).sum()
        with autocast_context:
            logits = model(input_ids)
            loss = criterion(input=logits.view(-1, logits.size(-1)), target=labels.view(-1))

        evaluation_loss.update(loss.detach(), num_items_in_batch)
        progress_bar.set_postfix({"loss": f"{loss:0.3f}"})
        progress_bar.update()
        if (batch_idx + 1) >= eval_steps:
            break

    # set model back to the original mode
    model.train(is_training)

    if args.ddp_enabled:
        evaluation_loss.reduce(dst=args.master_rank)

    progress_bar.close()

    return {
        "loss": evaluation_loss.average,
    }


def setup_ddp(args: argparse.Namespace) -> None:
    args.rank = int(os.environ.get("RANK", 0))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    args.ddp_enabled = os.environ.get("RANK", -1) != -1
    args.master_rank = 0
    args.local_master_rank = 0
    args.is_master = args.rank == args.master_rank
    args.is_local_master = args.local_rank == args.local_master_rank
    if args.ddp_enabled:
        # set appropriate CUDA device
        torch.cuda.set_device(args.local_rank)
        # init process group
        dist.init_process_group(backend=getattr(args, "ddp_backend", "nccl"))  # nccl, gloo, etc


def cleanup_ddp(args: argparse.Namespace) -> None:
    if args.ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
