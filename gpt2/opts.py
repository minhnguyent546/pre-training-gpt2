import argparse


def add_run_pretrain_opts(parser: argparse.ArgumentParser) -> None:
    """Options for pre-training model."""
    _add_general_opts(parser)
    _add_dataset_opts(parser)
    _add_model_opts(parser)
    _add_wandb_opts(parser)
    _add_common_training_opts(parser)
    _add_ddp_training_opts(parser)

def add_run_pretrain_xla_opts(parser: argparse.ArgumentParser) -> None:
    """Options for pre-training model with XLA devices."""
    _add_general_opts(parser)
    _add_dataset_opts(parser)
    _add_model_opts(parser)
    _add_wandb_opts(parser)
    _add_common_training_opts(parser)
    _add_xla_training_opts(parser)

def _add_general_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('General')
    group.add_argument(
        '--checkpoints-dir',
        type=str,
        help='Directory to save model checkpoints',
        default='checkpoints',
    )
    group.add_argument(
        '--seed',
        type=int,
        help='Seed for random number generators',
        default=1061109567,
    )

def _add_dataset_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Dataset')
    group.add_argument(
        '--train-dir',
        type=str,
        help='Directory contains training shards',
        default='fineweb_edu/train',
    )
    group.add_argument(
        '--valid-dir',
        type=str,
        help='Directory contains validation shards',
        default='fineweb_edu/valid',
    )
    group.add_argument(
        '--drop-last',
        help='Whether to drop the last incomplete batch',
        action='store_true',
    )

def _add_model_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--vocab-size',
        type=int,
        help='Vocabulary size',
        default=50257,
    )
    group.add_argument(
        '--seq-length',
        type=int,
        help='Maximum sequence length',
        default=512,
    )
    group.add_argument(
        '--d-model',
        type=int,
        help='Size of the embedding vectors',
        default=768,
    )
    group.add_argument(
        '--num-layers',
        type=int,
        help='Number of hidden layers',
        default=12,
    )
    group.add_argument(
        '--num-heads',
        type=int,
        help='Number of attention heads',
        default=12,
    )
    group.add_argument(
        '--d-ff',
        type=int,
        help='Intermediate size of the feed-forward layers',
        default=3072,
    )
    group.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate',
        default=0.0,
    )
    group.add_argument(
        '--activation',
        type=str,
        help='Which activation function to use',
        choices=['relu', 'gelu'],
        default='gelu',
    )
    group.add_argument(
        '--tie-weights',
        action='store_true',
        help='Whether to tie weights between input and output embeddings',
    )

def _add_wandb_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Wandb')
    group.add_argument(
        '--wandb-logging',
        action='store_true',
        help='Enable logging to wandb',
    )
    group.add_argument(
        '--wandb-project',
        type=str,
        help='Project name',
        default='pre-training-gpt2',
    )
    group.add_argument(
        '--wandb-name',
        type=str,
        help='Experiment name',
        default='base',
    )
    group.add_argument(
        '--wandb-logging-interval',
        type=int,
        help='Time between syncing metrics to wandb',
        default=500,
    )
    group.add_argument(
        '--wandb-resume-id',
        type=str,
        help='Id to resume a run from',
    )
    group.add_argument(
        '--wandb-notes',
        type=str,
        help='Wandb notes',
    )
    group.add_argument(
        '--wandb-tags',
        type=str,
        help='Wandb tags',
    )

def _add_common_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('Training')
    group.add_argument(
        '--compile',
        help='Whether to compile the model with torch.compile',
        action='store_true',
    )
    group.add_argument(
        '--matmul-precision',
        type=str,
        help='Sets the internal precision of float32 matrix multiplications',
        choices=['highest', 'high', 'medium'],
        default='highest',
    )

    # optimizer options
    group.add_argument(
        '--optim-type',
        type=str,
        help='Which optimizer to use',
        choices=['adam', 'adamw'],
        default='adamw',
    )
    group.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate',
        default=6.0e-4,
    )
    group.add_argument(
        '--betas',
        nargs=2,
        type=float,
        help='Optimizer beta values',
        default=[0.9, 0.999],
    )
    group.add_argument(
        '--weight-decay',
        type=float,
        help='Weight decay value',
        default=0.0,
    )

    # scheduler options
    group.add_argument(
        '--decay-method',
        type=str,
        help='Learning rate decay method (you might want to choose larger learning rate when using noam decay, e.g. 0.5)',
        choices=['cosine', 'noam'],
        default='cosine',
    )
    group.add_argument(
        '--warmup-steps',
        type=int,
        help='Warmup steps for learning rate',
        default=1_000,
    )
    group.add_argument(
        '--min-lr',
        type=float,
        help='Minimum learning rate (i.e. decay until this value) (for noam decay only)',
        default=6.0e-5,
    )
    group.add_argument(
        '--decay-steps',
        type=int,
        help='Number of steps to decay learning rate (for cosine decay only)',
        default=20_000,
    )

    # others
    group.add_argument(
        '--train-batch-size',
        type=int,
        help='Training batch size',
        default=32,
    )
    group.add_argument(
        '--eval-batch-size',
        type=int,
        help='Evaluation batch size',
        default=32,
    )
    group.add_argument(
        '--gradient-accum-step',
        type=int,
        help='Gradient accumulation step',
        default=1,
    )
    group.add_argument(
        '--mixed-precision',
        type=str,
        help='Data type for mixed precision training',
        choices=['fp16', 'bf16'],
    )
    group.add_argument(
        '--train-steps',
        type=int,
        help='Number of training steps (i.e. number of optimizer steps)',
        default=20_000,
    )
    group.add_argument(
        '--valid-steps',
        type=int,
        help='Number of validation steps',
        default=100,
    )
    group.add_argument(
        '--valid-interval',
        type=int,
        help='Steps between validation',
        default=1_000,
    )
    group.add_argument(
        '--save-interval',
        type=int,
        help='Steps between saving checkpoints (you SHOULD use the SAME value as --valid-interval for accurate training loss when resuming from previous checkpoint)',
        default=1_000,
    )
    group.add_argument(
        '--saved-checkpoint-limit',
        type=int,
        help='Maximum number of saved checkpoints, when reached, the oldest checkpoints will be removed',
        default=10,
    )
    group.add_argument(
        '--max-grad-norm',
        type=float,
        help='Maximum gradient norm for gradient clipping (0.0 means no clipping)',
        default=0.0,
    )
    group.add_argument(
        '--from-checkpoint',
        type=str,
        help='Start training from this checkpoint, use, e.g. gpt2 or gpt2-large for the pretrained model',
    )

def _add_ddp_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('DDP training')
    group.add_argument(
        '--ddp-backend',
        type=str,
        help='DDP backend used for distributed training',
        default='nccl',
    )

def _add_xla_training_opts(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('XLA training')
    group.add_argument(
        '--mp-start-method',
        type=str,
        choices=['fork', 'spawn'],
        help='Multiprocessing start method',
        default='fork',
    )
    group.add_argument(
        '--ddp',
        help='Use distributed data parallel for gradient reducing',
        action='store_true',
    )
    group.add_argument(
        '--use-syncfree-optim',
        help='Use sync-free optimizer version for better performance when using mixed precision training',
        action='store_true',
    )
