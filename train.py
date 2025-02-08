import dataclasses
import os
from pathlib import Path

import torch
import wandb
from dargparser import dargparse
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import (
    LightningEnvironment,
    SLURMEnvironment,
)
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
from transformers import PreTrainedModel

from numerical_table_questions.arguments import TrainingArgs, MiscArgs, TokenizationArgs
from numerical_table_questions.utils.dlib.frameworks.lightning import CUDAMetricsCallback
from numerical_table_questions.utils.dlib.frameworks.pytorch import get_rank, set_torch_file_sharing_strategy_to_system
from numerical_table_questions.utils.dlib.frameworks.wandb import (
    WANDB_ENTITY,
    WANDB_PROJECT,
    WandbCleanupDiskAndCloudSpaceCallback,
    check_checkpoint_path_for_wandb,
    check_for_wandb_checkpoint_and_download_if_necessary,
)
from numerical_table_questions.data_loading import TableQADataModule
from numerical_table_questions.utils.system_helpers import (
    handle_batch_size_logic_,
    log_slurm_info,
    parse_auto_arguments,
)
from numerical_table_questions.model import LightningWrapper
from numerical_table_questions.utils.model_utils import get_model_module, get_model_specific_config


@logger.catch(reraise=True)
def main(parsed_arg_groups: tuple[TrainingArgs, MiscArgs, TokenizationArgs]):
    current_process_rank = get_rank()
    args, misc_args, tokenizer_args = parsed_arg_groups

    ################ Apply fixes ##############
    if misc_args.too_many_open_files_fix:
        logger.info("Setting torch sharing strategy to 'file_system'")
        set_torch_file_sharing_strategy_to_system()

    ############# Seed ##############
    misc_args.seed = seed_everything(seed=misc_args.seed, workers=True)

    ############# Construct W&B Logger ##############
    if misc_args.offline or misc_args.fast_dev_run or args.data_preprocessing_only:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_extra_args = dict(
        name=misc_args.wandb_run_name,
    )
    if (
        args.checkpoint_path
        and args.resume_training
        and check_checkpoint_path_for_wandb(args.checkpoint_path)
    ):
        logger.info("Resuming training from W&B")
        wandb_extra_args = dict(
            id=check_checkpoint_path_for_wandb(args.checkpoint_path), resume="must"
        )  # resume W&B run
    else:
        args.resume_training = False

    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=misc_args.wandb_tags,
        **wandb_extra_args,
    )

    parse_auto_arguments(args)
    effective_batch_size_per_step = handle_batch_size_logic_(args)

    ########### Log config ###########
    for arg_group in parsed_arg_groups:
        wandb_logger.log_hyperparams(dataclasses.asdict(arg_group))
        if current_process_rank == 0:
            logger.info(arg_group)

    if current_process_rank == 0 and not args.resume_training and not misc_args.offline:
        if misc_args.wandb_run_name is None:
            logger.warning(
                "No run name specified with `--wandb_run_name`. Using W&B default (randomly generated name)."
            )
        else:
            wandb_logger.experiment.name = (
                misc_args.wandb_run_name + "-" + wandb_logger.version
            )  # Append id to name for easier recognition in W&B UI

    IS_ON_SLURM = SLURMEnvironment.detect()
    if IS_ON_SLURM and current_process_rank == 0:
        log_slurm_info()
    ########### Calulate training constants ###########

    if args.training_goal_unit == "samples":
        goal_units_per_optimizer_step = args.effective_batch_size
        goal_units_per_forward_pass = effective_batch_size_per_step
    elif args.training_goal_unit == "tokens":
        goal_units_per_optimizer_step = args.effective_batch_size * args.max_sequence_length
        goal_units_per_forward_pass = effective_batch_size_per_step * args.max_sequence_length
    elif args.training_goal_unit == "optimizer-steps":
        goal_units_per_optimizer_step = 1
        goal_units_per_forward_pass = 1 / args.gradient_accumulation_steps
    else:
        raise ValueError(f"Unknown training goal unit: {args.training_goal_unit}")

    # Lightning does `gradient_accumulation_steps` many forward passes per trainer step (step := optimization step)
    args.training_goal = int(args.training_goal / goal_units_per_optimizer_step)
    val_frequency_in_optimization_steps = int(args.val_frequency / goal_units_per_optimizer_step)

    # val_frequency in lightning is every forward pass NOT optimization step # NOTE: as of June 2023
    args.val_frequency = int(args.val_frequency / goal_units_per_forward_pass)
    args.model_log_frequency = int(args.model_log_frequency / goal_units_per_optimizer_step)
    args.lr_warmup = int(args.lr_warmup / goal_units_per_optimizer_step)

    checkpoint_callback = ModelCheckpoint(
        filename="snap-{step}-samples-{progress/samples}-{progress/tokens}-loss-{val/loss:.2f}",
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        every_n_train_steps=args.model_log_frequency,
    )
    if args.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=0.00,
            patience=args.early_stopping_patience,
            mode="min",
            )
    wandb_disk_cleanup_callback = WandbCleanupDiskAndCloudSpaceCallback(
        cleanup_local=True, cleanup_online=False, size_limit=20
    )

    ################# Construct model ##############

    # process (download) checkpoint_path to enable remote wandb checkpoint paths
    if args.checkpoint_path:
        args.checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            args.checkpoint_path, wandb_logger.experiment
        )
    # Resume (interrupted) training run from checkpoint if specified (if condition is True)
    if args.resume_training and args.checkpoint_path:  # load weights, optimizer states, scheduler state, ...\
        model = LightningWrapper.load_from_checkpoint(
            args.checkpoint_path,
            model=get_model_module(args.model_name_or_path),
            effective_batch_size_per_step=effective_batch_size_per_step,
        )
        logger.info(f"Loded model with {model.samples_processed.item()} processed samples from checkpoint. "
                    " Also loaded all training states - ready to resume training."
                    )
    else:  # create new model
        model_module = get_model_module(args.model_name_or_path)
        model_config = get_model_specific_config(args.model_name_or_path)
        model = LightningWrapper(model_module, args, **model_config, effective_batch_size_per_step=effective_batch_size_per_step)

    # initiallize from pretrained but do not resume training, instead fresh start (if condition is True)
    if args.checkpoint_path and not args.resume_training:
        # load only weights but nooptimizer states, etc.
        torch_load = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(torch_load["state_dict"], strict=False)
        # reset for fresh training run
        model.samples_processed = torch.tensor(0.0)
        model.tokens_processed = torch.tensor(0.0)

    if args.train_only_embeddings:
        if get_rank() == 0:
            logger.info("Training only embedding layer")
        for param in model.model.parameters():
            param.requires_grad = False
        model.model.get_input_embeddings().weight.requires_grad = True

    if args.from_scratch_embeddings:
        torch.nn.init.xavier_uniform_(model.model.get_input_embeddings().weight)
        # torch.nn.init.normal_(model.model.get_input_embeddings().weight) # alternative

    if current_process_rank == 0:
        model.on_train_start = lambda: logger.info(
            f"Total optimizer steps: {args.training_goal} | "
            f"LR warmup steps: {args.lr_warmup} | "
            f"Validation Frequency: {val_frequency_in_optimization_steps} | "
            f"Model Log Frequencey: {args.model_log_frequency} | "
            f"Effective batch size: {args.effective_batch_size}"
        )
    wandb_logger.watch(model, log="gradients", log_freq=500, log_graph=False)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."  # noqa: E501
                "Please install torch >= 2.0 or disable compile."
            )
        # only compile the model wrapped within the Lightning Module since wandb logging
        # (used in the train/val/... loops) is currently incompatible with compile in torch 2.1
        model.model = torch.compile(model.model)

    #################### Construct dataloaders & trainer #################
    dm = TableQADataModule(model.model_specs,
                           table_corpus=args.table_corpus_name,
                           dataset_name=args.dataset_suffix,
                           train_batch_size=args.batch_size_per_device,
                           eval_batch_size=args.eval_batch_size_per_device,
                           lazy_data_processing=args.lazy_data_processing,
                           is_batch_dict=args.is_batch_dict,
                           data_dir=args.data_dir,
                           tokenizing_args=tokenizer_args,
                           num_dataloader_workers=args.workers,
                           too_many_open_files_fix=misc_args.too_many_open_files_fix,
                           )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, wandb_disk_cleanup_callback, lr_monitor]
    if args.early_stopping_patience > 0:
        callbacks.append(early_stop_callback)
    if args.accelerator == "cuda":
        callbacks.append(CUDAMetricsCallback())

    # "smart" DDP skipping the find_unused_parameters step - slightly faster
    distributed_strategy = (
        DDPStrategy(find_unused_parameters=False)
        if args.accelerator == "cuda" and args.distributed_strategy == "ddp_smart"
        else args.distributed_strategy
    )

    plugins = None
    if IS_ON_SLURM:
        logger.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
        plugins = [LightningEnvironment()]

    # Initialize trainer
    trainer = Trainer(
        max_steps=args.training_goal,
        val_check_interval=args.val_frequency,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=args.cuda_device_ids or args.num_devices,
        accelerator=args.accelerator,
        strategy=distributed_strategy,
        logger=wandb_logger,
        deterministic=misc_args.force_deterministic,
        callbacks=callbacks,
        plugins=plugins,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        fast_dev_run=misc_args.fast_dev_run,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other  # noqa: E501
    )

    if args.val_before_training and not args.resume_training:
        # TODO: we could use a new trainer with Trainer(devices=1, num_nodes=1) to prevent samples from possibly getting replicated with DistributedSampler here.  # noqa: E501
        logger.info(f"Rank {current_process_rank} | Validation before training...")
        # somehow the dm initialization does not happen implicitly, maybe only when trainer.fit is called?
        dm.prepare_data()
        dm.setup('fit')
        val_result = trainer.validate(model, datamodule=dm)
        logger.info(f"Validation Result: {val_result}.")
        print(val_result)
        # clear results because reuse of validate might cause problems otherwise https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        trainer.validate_loop._results.clear()
        if args.val_only:
            exit(0)

    logger.info(f"Rank {current_process_rank} | Starting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint_path if args.resume_training else None)
    if trainer.interrupted and IS_ON_SLURM:
        logger.error(
            "Detected keyboard interrupt, not trying to save latest checkpoint right now because we detected SLURM and do not want to drain the node..."
        )
    else:
        logger.success("Fit complete, starting validation...")
        # Validate after training has finished
        trainer.validate(model, datamodule=dm)

        if current_process_rank == 0:
            logger.info("Trying to save checkpoint....")

            save_path = str(Path(checkpoint_callback.dirpath) / "last_model_ckpt.ckpt")
            trainer.save_checkpoint(save_path)

            logger.info("Collecting PL checkpoint for wandb...")
            artifact = wandb.Artifact(name=f"model-{wandb_logger.experiment.id}", type="model")
            artifact.add_file(save_path, name="model.ckpt")

            logger.info('Collecting "raw" HF checkpoint for wandb...')
            # Also save raw huggingface checkpoint, so that we don't need lightning and the current code structure to load the weights  # noqa: E501
            raw_huggingface_model: PreTrainedModel = trainer.lightning_module.model
            save_path = str(Path(checkpoint_callback.dirpath) / "raw_huggingface")
            raw_huggingface_model.save_pretrained(save_path)
            artifact.add_dir(save_path, name="raw_huggingface")

            logger.info("Pushing to wandb...")
            aliases = ["train_end", "latest"]
            wandb_logger.experiment.log_artifact(artifact, aliases=aliases)

            logger.success("Saving finished!")

        if args.test_after_train_end:
            if torch.cuda.device_count() > 1 and current_process_rank == 0:
                logger.info(f"Training was performed on {args.num_devices} devices. "
                            "Testing should be run on a single device if possible. Skip testing for now (use evaluate.py).")
            else:
                logger.info("Testing model performance...")
                trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    parsed_arg_groups = dargparse(dataclasses=(TrainingArgs, MiscArgs, TokenizationArgs))
    main(parsed_arg_groups)
