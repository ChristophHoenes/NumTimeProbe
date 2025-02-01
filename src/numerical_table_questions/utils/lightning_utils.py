from lightning.pytorch import Trainer


def get_lightning_trainer(args, wandb_logger=None, deterministic: bool = True) -> Trainer:
    trainer = Trainer(
        max_steps=args.training_goal,
        val_check_interval=args.val_frequency,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=args.cuda_device_ids or args.num_devices,
        accelerator=args.accelerator,
        logger=wandb_logger,
        deterministic=deterministic,
        precision=args.precision,
        gradient_clip_val=args.gradient_clipping,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        inference_mode=not args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other  # noqa: E501
    )
    return trainer
