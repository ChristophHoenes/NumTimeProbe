from typing import Dict, Union

from lightning.pytorch import Trainer


def disambiguate_val_frequency_for_lightning(args) -> Dict[str, Union[int, float]]:
    if args.val_frequency_per_epoch is not None:
        return {"val_check_interval": 1.0 / args.val_frequency_per_epoch if args.val_frequency_per_epoch >= 1 else None,
                'check_val_every_n_epoch': int(1/args.val_frequency_per_epoch) if args.val_frequency_per_epoch < 1 else None
                }
    elif args.val_frequency_goal_unit is not None:
        return {"val_check_interval": int(args.val_frequency_goal_unit),
                'check_val_every_n_epoch': None
                }
    else:
        raise ValueError("Exactly one of val_frequency_per_epoch or val_frequency_goal_unit must be set.")


def get_lightning_trainer(args, wandb_logger=None, deterministic: bool = True) -> Trainer:
    val_frequency_args = disambiguate_val_frequency_for_lightning(args)
    trainer = Trainer(
        max_steps=args.training_goal,
        **val_frequency_args,
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
