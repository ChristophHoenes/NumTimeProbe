import warnings
from typing import List, Union

import torch
from lightning.pytorch.loggers import WandbLogger

from numerical_table_questions.utils.dlib.frameworks.wandb import WANDB_ENTITY, WANDB_PROJECT, check_for_wandb_checkpoint_and_download_if_necessary


def try_load_local_then_wandb_checkpoint(model, wandb_logger, checkpoint_path: str, map_location: str = "cpu"):
    # first check if the checkpoint is available locally to avoid unnecessary downloading
    try:
        checkpoint_content = torch.load(checkpoint_path, map_location=torch.device(map_location))
    except FileNotFoundError:
        warnings.warn(f"Checkpoint file '{checkpoint_path}' not found locally. Trying to download it from W&B...")
        # try download remote wandb checkpoint for provided path/model id
        resolved_checkpoint_path = check_for_wandb_checkpoint_and_download_if_necessary(
            checkpoint_path, wandb_logger.experiment
            )
        checkpoint_content = torch.load(resolved_checkpoint_path)
    model.load_state_dict(checkpoint_content["state_dict"])


def get_wandb_logger(misc_args, tags: Union[str, List[str]] = []) -> WandbLogger:
    if isinstance(tags, str):
        tags = [tags]
    return WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=[*tags, *misc_args.wandb_tags],
        name=misc_args.wandb_run_name,
    )
