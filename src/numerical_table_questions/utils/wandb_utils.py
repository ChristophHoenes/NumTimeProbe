import wandb
import warnings
from pathlib import Path
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


def get_artifact(run_id: str, artifact_name: str, version_tag: str = 'latest', return_local_path: bool = False) -> Union[wandb.Artifact, List[str]]:
    api = wandb.Api()
    if artifact_name == 'model':
        artifact_name = f"model-{run_id}:{version_tag}"
    else:
        artifact_name = f"run-{run_id}-{artifact_name}:{version_tag}"
    artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}")
    if return_local_path:
        local_path = Path.cwd() / 'artifacts' / artifact_name
        if local_path.is_dir():
            return [str(file) for file in local_path.iterdir()]
        else:
            filepath_string = artifact.download()
            return [str(file) for file in Path(filepath_string).iterdir()]
    return artifact


def get_run(run_id: str):
    api = wandb.Api()
    return api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
