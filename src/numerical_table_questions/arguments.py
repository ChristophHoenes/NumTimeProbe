import os
from dataclasses import dataclass
from dargparser import dArg
from typing import Literal

from loguru import logger
from torch import multiprocessing


@dataclass
class TrainingArgs:
    model_name_or_path: str = dArg(
        default="tapex",
        help="HuggingFace model identifier. This is used to construct the model architecture and load pretrained weights if not specified otherwise.",  # noqa: E501
        aliases="--model",
    )
    language_modeling_strategy: Literal["mlm", "clm"] = dArg(
        default="mlm",
        help="Whether to train a masked language model or a causal language model.",
    )
    resume_training: bool = dArg(
        default=False,
        help="Whether to resume training from checkpoint or only load the weights. If true, `--checkpoint_path` must be specified.",
        aliases="--resume",
    )
    checkpoint_path: str | None = dArg(
        default=None,
        help="Path to a saved pytorch-lightning checkpoint. Use the wandb:<wandb-run-id> syntax to load a checkpoint from W&B.",  # noqa: E501
        aliases="--checkpoint",
    )
    tokenizer_path: str | None = dArg(
        default=None,
        help="Path to a directory containing a saved Huggingface PreTrainedTokenizer.",
        aliases="--tokenizer",
    )
    data_dir: str = dArg(
        default="./data",
        help="Path to the data directory. By default, expects a train.txt and dev.txt file inside the directory.",  # noqa: E501
        aliases="-d",
    )
    train_file: str = dArg(default="train.txt")
    dev_file: str = dArg(default="dev.txt")
    line_by_line: bool = dArg(
        default=False, help="Process dataset line by line instead of chunking."
    )
    language: str = dArg(
        default=None,
        help="If specified, the data is expected to lie inside a subdirectory with this name.",
        aliases=["--lang", "--lg", "-l"],
    )
    max_sequence_length: int = dArg(
        default=1024,  # this is the case for tapex but 512 is more common?
        help="Sequence length for dataset tokenization.",
        aliases=["--seq_len", "--block_size"],
    )
    overwrite_data_cache: bool = dArg(
        default=False, help="Overwrite the cached preprocessed datasets or not.", aliases="--odc"
    )
    conserve_disk_space: bool = dArg(
        default=False, help="Cleanup cache files whenever possible to save disk space."
    )
    data_preprocessing_only: bool = dArg(
        default=False, help="Exit the script after data preprocessing. Do not start training."
    )
    force_test_loss_computation: bool = dArg(
        default=False,
        help=("Whether to always compute a forward pass during testing to obtain the test loss. "
              "Sometimes alternative metrics (that use the generate instead of the forward method) "
              "are more interesting than the loss and the time of computing forward can be saved."),
    )
    test_after_train_end: bool = dArg(
        default=True,
        help=("Whether to run a test epoch directly after training or not."),
    )

    ####### Hardware ###########
    accelerator: Literal["cuda", "cpu", "tpu", "mps", "auto"] = dArg(
        default="auto",
        help='Hardware accelerator to use. If "auto", will auto-detect available hardware accelerator.',
    )
    distributed_strategy: Literal[
        "ddp", "fsdp", "ddp_smart", "ddp_spawn", "ddp_fork", "auto"
    ] = dArg(
        default="auto",
        help="Distributed training strategy to use. If `auto`, will select automatically (no distributed strategy is used when using a single device).",
        aliases="--ds",
    )
    num_devices: int = dArg(
        default=-1,
        aliases=["--devices", "--nd"],
        help="Number of devices to use for distributed training. If -1, will use all available devices (CUDA) or an accelerator-specific default.",
    )
    cuda_device_ids: list[int] = dArg(
        default=[],
        aliases="--gpu_ids",
        help="Specific CUDA devices (selected by specified indices) to use. Overwrites `--num_devices`. Requires CUDA on the host system.",
    )
    workers: int = dArg(
        default=4,
        help="Number of workers for dataloaders. *Every device* will use that many workers.",
        aliases="-w",
    )
    preprocessing_workers: int = dArg(
        default=-1,
        help="Number of workers for preprocessing the datasets. If -1, use all available CPUs.",
        aliases="--pw",
    )
    precision: Literal["16-mixed", "bf16-mixed", 32] = dArg(
        default=32,
        help="Floating point precision to use during training. Might require specific hardware.",
    )
    compile: bool = dArg(
        default=False,
        help="Whether to compile the model with `torch.compile`. Requires torch>=2.0",
    )

    ####### General training ###########
    training_goal: int = dArg(
        default=200_000, help="Number training goal units to train for.", aliases="--tg"
    )
    training_goal_unit: Literal["samples", "tokens", "optimizer-steps"] = dArg(
        default="samples", help="Unit of training_goal."
    )
    val_frequency: float = dArg(
        default=0.05,
        help="Period in training goal units between two validations. If <1, compute as fraction of training_goal",
        aliases="--vfq",
    )
    model_log_frequency: float = dArg(
        default=0.1,
        help="Period in training goal units between two model checkpoints. If <1, compute as fraction of training_goal",
        aliases="--mfq",
    )
    val_before_training: bool = dArg(default=True, help="Run one validation epoch before training.")
    val_only: bool = dArg(default=False, help="Run one validation epoch before training.")
    batch_size_per_device: int = dArg(
        default=16,
        help="Batch size per device. If effective_batch_size is specified, this is the maximum batch size per device (you should then increase this in powers of two until you get CUDA OOM errors).",  # noqa: E501
        aliases="-b",
    )
    eval_batch_size_per_device: int = dArg(
        default=32,
        help="Batch size per device for evaluation (no gradients -> can be larger than batch_size_per_device for training).",
    )
    effective_batch_size: int | None = dArg(
        default=None,
        help="If set, try to auto-infer batch_size_per_device and gradient_accumulation_steps based on number of devices given by --num_devices.",  # noqa: E501
        aliases=["--eb"],
    )
    lazy_data_processing: bool = dArg(
        default=True,
        help=("If True (default) will execute data processing (e.g tokenization) during data loading. "
              "Else will process the whole dataset before the main loop and save it to disk (might consume a lot of disk space!))."
              ),
    )
    is_batch_dict: bool = dArg(
        default=True,
        help=("If True (default) will data loader will return batch as dict. Else will return tuple of tensors."),
    )
    learning_rate: float = dArg(default=5e-5, aliases="--lr")
    lr_warmup: float = dArg(
        default=0.1,
        help="Number of training goal units to do a learning rate warmup. If <1, compute as fraction of training_goal.",  # noqa: E501
    )
    lr_schedule: Literal[
        "cosine", "linear", "reduce_on_plateau", "constant", "cosine_with_restarts", "polynomial"
    ] = dArg(default="cosine", help="Learning rate schedule.")
    weight_decay: float = dArg(default=0.0, aliases="--wd")
    gradient_clipping: float | None = dArg(default=None, aliases="--gc")
    gradient_accumulation_steps: int = dArg(default=1, aliases=["--gas", "--accum"])
    train_only_embeddings: bool = dArg(
        default=False,
        help="Train only the embedding layer of the model and keep the other transformer layers frozen.",  # noqa: E501
        aliases="--only_embeddings",
    )
    from_scratch: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the model."
    )
    from_scratch_embeddings: bool = dArg(
        default=False, help="Do not use pre-trained weights to intialize the token embeddings."
    )
    table_corpus: str = dArg(
        default="wikitables", help="Name of the table corpus the dataset is based on."
    )
    dataset_name: str = dArg(
        default="basic_dataset", help="Name of the dataset to use."
    )
    dummy_ipykernel_fix: str = dArg(
        default='',
        help="flag --f with mamba path is passes automatically, if unknown argument throws error in Notebooks.",
        aliases=["--f"]
    )
    only_first_x_tokens: int = dArg(
        default=0,
        help=("If > 0 overwrites the input with padding after only_first_x_tokens tokens. "
              "This can be used to check if the model only remembers the results from training "
              "by recognizing markers in the beginning of the input (e.g query only, ignoring input)."),
        aliases=["--only_query"],
    )

    def __post_init__(self):
        if self.val_frequency < 1:
            self.val_frequency = int(self.training_goal * self.val_frequency)
        if self.model_log_frequency < 1:
            self.model_log_frequency = int(self.training_goal * self.model_log_frequency)
        if self.lr_warmup < 1:
            self.lr_warmup = int(self.training_goal * self.lr_warmup)
        if self.cuda_device_ids:
            if self.num_devices != -1:
                logger.warning(
                    f"Overwriting --num_devices={self.num_devices} with {len(self.cuda_device_ids)} because of --cuda_device_ids={self.cuda_device_ids}"
                )
            self.num_devices = len(self.cuda_device_ids)
        if self.preprocessing_workers == -1:
            # Set to all available CPUs, handle SLURM case when only some CPUs are available to the job
            self.preprocessing_workers = int(
                os.environ.get("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count())
            )


@dataclass
class MiscArgs:
    seed: int | None = None
    force_deterministic: bool = dArg(
        default=False, help="Force PyTorch operations to be deterministic."
    )
    offline: bool = dArg(default=False, help="Disable W&B online syncing.")
    fast_dev_run: bool = dArg(
        default=False, help="Do fast run through training and validation with reduced sizes."
    )
    wandb_run_name: str | None = dArg(
        default=None, help="Run name for the W&B online UI.", aliases="-n"
    )
    wandb_tags: list[str] = dArg(default=[])
    wandb_project: str = dArg(default=None)
    too_many_open_files_fix: bool = dArg(
        default=False,
        help='Apply fix to circumvent "Too many open files" error caused by the PyTorch Dataloader when using many workers or large batches.',  # noqa: E501
        aliases="--open_files_fix",
    )


@dataclass
class TokenizationArgs:
    allow_custom_truncation: bool = dArg(
        default=False,
        help=("Whether special tokenizer truncation arguments should be allowed or not (default is Fales = not allowed)."),
    )
    query_first: bool = dArg(
        default=True,
        help=("Whether the question is expected to be at the beginning (default: True) or "
              "the end of the input (if set to False)."),
    )
    padding: Literal['longest', 'do_not_pad', 'max_length', 'True', 'False'] = dArg(
        default='max_length',
        help="Whether to pad all sequences to the longest provided sequence length, the max_length or not at all (default).",
    )
    truncation: Literal['longest_first', 'do_not_truncate', 'only_first', 'only_second', 'True', 'False', 'drop_rows_to_fit'] = dArg(
        default='True',
        help=("Whether to truncate sequences that are longer than max_lenghth. "
              "For explanations of the options plese consult the documentation of the tokenizer in use."
              "'only_first' and 'only_second' are applicable if sequences are provided as pairs "
              "and only truncate one of the sequences for each sample."),
    )
    keep_oversized_samples: bool = dArg(
        default=False,
        help=("Whether or not to keep examples that do not fit into memory. If False, too long sequences are filtered (default), "
              "requires truncation 'do_not_truncate' or 'False'."),
    )
    optimize_int_type: bool = dArg(
        default=True,
        help="Whether or not to convert integers to the smallest possible int dtype to conserve disk space.",
    )
    return_tensors: Literal['pt', 'tf', 'np'] = dArg(
        default='pt',
        help="Whether to return tensors as PyTorch = 'pt' (default), Tensorflow = 'tf' or numpy = 'np' type.",
        aliases="--tensor_format",
    )

    def __post_init__(self):
        # convert strings 'True' and 'False' to bool for padding and truncation
        # circumvents only one type for each argument constraint of dArg
        if self.padding == 'True':
            self.padding = True
        elif self.padding == 'False':
            self.padding = False

        if self.truncation == 'True':
            self.truncation = True
        elif self.truncation == 'False':
            self.truncation = False

        # check if truncation is set to a valid value in combination with keep vs. filter too long samples option
        reducing_truncation = [True, 'longest_first']
        if self.keep_oversized_samples and self.truncation not in reducing_truncation:
            raise ValueError("If samples that do not fit into the model should be kept the truncation strategy must reduce the samples size accordingly! "
                             f"Current options are {reducing_truncation} but selected truncation is {self.truncation}."
                             )


@dataclass
class DataProcessingArgs:
    data_dir: str = dArg(
        default="./data/NumTabQA/.cache", help="File path to the location in the file system where the data is stored."
    )
    table_corpus: str = dArg(
        default="wikitables", help="Name of the table corpus the dataset is based on."
    )
    dataset_name: str = dArg(
        default="basic_dataset", help="Name of the dataset to use."
    )
    split: str = dArg(
        default="test", help="Name of the split to use."
    )
