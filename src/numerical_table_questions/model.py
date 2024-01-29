from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import lightning as L
import torch
import transformers
from loguru import logger
from torch.optim import AdamW
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler

from dlib.frameworks.pytorch import get_rank

if TYPE_CHECKING:
    from train import TrainingArgs


@dataclass
class OptimizerArgs:
    optimizer_class: torch.optim.Optimizer = AdamW
    kwargs: dict = field(
        default_factory=lambda: {
            'lr': 1e-3,
            'betas': (0.9, 0.999),
            'epsilon': 1e-8
        }
    )


@dataclass
class ModelTypeInfo:
    model_name_or_path: str
    pad_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None
    # index of loss in outputs
    loss_out_id: Optional[Union[int, str]] = None
    prediction_scores_out_id: Optional[Union[int, str]] = 0
    hidden_state_out_id: Optional[Union[int, str]] = None
    attention_out_id: Optional[Union[int, str]] = None
    # whether the forward pass expects the tragets or not
    input_targets: bool = False
    # any additional model specific arguments
    # (if key is of type int it is interpreted as positional or otherwise as keyword argument)
    input_mapping: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO value checks
        pass


def order_positional_arguments(inputs, target, input_map: dict) -> tuple:
    # this function has side effects by altering input_map
    # integer keys in input map are used to process positional embeddings
    integer_keys = [key for key in input_map if isinstance(key,  int)]
    # on first call parse '*' syntax sugar for positional arguments
    # (avoids having to list complete argument order explicitly on ModelTypeInfo initialization)
    if '*' in input_map:
        parsed_input_map = dict()
        # positional arguments that are placed to a specific position of the model inputs and
        # are not part of the inputs provided by the data loader
        injection_positions = integer_keys  # use alias for better readability (understanding)
        # 'injection_positions or [0]' prevents error from taking max over an empty list
        # max(injection_positions or [0]) == 0 if injection_positions == []
        if max(injection_positions or [0]) >= len(inputs) + len(injection_positions):
            raise ValueError(f"Found value {max(injection_positions or [0])} in input_map keys which exceeds "
                             f"the expected lengths ({len(inputs) + len(injection_positions)}) of positional arguments!")
        # inject explicit fills as positional arguments in correct order
        inject_count = 0
        for i in range(len(inputs) + len(injection_positions)):
            if i in injection_positions:
                # inject argument at specified position
                # (the argument is always obtained by evaluating a (lambda) function)
                parsed_input_map[i] = input_map[i]
                # counter for injected arguments
                inject_count += 1
            else:
                # shift position of argument to the back by the number of preceeding injected positional arguments
                parsed_input_map[i] = lambda x, y, idx=i-inject_count: x[idx]
        input_map.update(parsed_input_map)
        del input_map['*']
    elif len(integer_keys) > 0:
        if sorted(integer_keys) != list(range(len(integer_keys))):
            raise ValueError(f"Integer keys in input_map must be consecutive range but are {integer_keys} instead!")
    # could be optimized by pre-computing order and just applying it here
    forward_args = sorted(
        [(model_input_id,
          inputs[data_input] if isinstance(data_input, int)
          else data_input(inputs, target)
          )
         for model_input_id, data_input in input_map.items()
         if isinstance(model_input_id, int)
         ]
        )
    return [tup[1] for tup in forward_args]


class LightningWrapper(L.LightningModule):
    def __init__(self,
                 model,
                 training_args: "TrainingArgs",  # do in string to remove dependency when loading.
                 model_type_info: Optional[ModelTypeInfo] = None,
                 loss_fn=torch.nn.functional.nll_loss,
                 optimizer_args: OptimizerArgs = OptimizerArgs(),
                 effective_batch_size_per_step=-100_000,
                 samples_processed: int = 0,
                 tokens_processed: int = 0):
        super().__init__()
        self.model = model
        self.model_specs = model_type_info or ModelTypeInfo(training_args.model_name_or_path)
        self.loss_fn = loss_fn
        self.optimizer_args = optimizer_args
        self.args = training_args
        self.effective_batch_size_per_step = effective_batch_size_per_step
        self.register_buffer("samples_processed", torch.tensor(samples_processed))
        self.register_buffer("tokens_processed", torch.tensor(tokens_processed))

    def forward(self, inputs, target=None):
        raise NotImplementedError("This is a break for preprocessing only. Remove this before training/validation.")
        if self.model_specs.input_targets and target is None:
            raise ValueError("Configuration argument 'input_targets' is set to True but no targets were provided as argument! "
                             "Please call the model forward pass with inputs and tragets.")
        # always wrap inputs in tuple
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        elif not isinstance(inputs, tuple):
            inputs = (inputs,)
        # use input order specified in model_specs
        if len(self.model_specs.input_mapping) > 0:
            # compute input format specified via self.model_specs.input_mapping
            forward_args = order_positional_arguments(inputs, target, input_map=self.model_specs.input_mapping)
            forward_kwargs = {model_input_id: data_function(inputs, target)
                              for model_input_id, data_function in self.model_specs.input_mapping.items()
                              if isinstance(model_input_id, str)
                              }
            outputs = self.model(*forward_args, **forward_kwargs)
        elif self.model_specs.input_targets:
            outputs = self.model(*inputs, target)
        else:
            outputs = self.model(*inputs)
        # always wrap outputs in tuple
        if isinstance(outputs, dict):
            outputs = tuple(outputs.values())
        elif not isinstance(outputs, tuple):
            outputs = (outputs,)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs, target)
        if self.model_specs.loss_out_id is None:
            # compute loss
            predictions = outputs[self.model_specs.prediction_scores_out_id]
            loss = self.loss_fn(predictions, target.view_as(predictions))
        else:
            # retrieve loss
            loss = outputs[self.model_specs.loss_out_id]
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        self.samples_processed += self.effective_batch_size_per_step
        self.tokens_processed += self.effective_batch_size_per_step * self.args.max_sequence_length
        self.log_dict(
            {
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs, target)
        if self.model_specs.loss_out_id is None:
            # compute loss
            loss = self.loss_fn(outputs[self.model_specs.prediction_scores_out_id], target.view(-1))
        else:
            # retrieve loss
            loss = outputs[self.model_specs.loss_out_id]

        self.log_dict(
            {
                "val/loss": loss.item(),
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay} and warmup steps: {self.args.lr_warmup}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        named_parameters = list(
            filter(lambda named_param: named_param[1].requires_grad, named_parameters)
        )

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.optimizer_args['optimizer_class'](optimizer_parameters,
                                                           **self.optimizer_args['kwargs'])

        if self.args.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, verbose=True
            )
            if self.args.lr_warmup > 0:  # Wrap ReduceLROnPlateau to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.args.lr_warmup,
                    after_scheduler=scheduler,
                )
            scheduler_config = {"frequency": self.args.val_frequency, "monitor": "train/loss"}
        else:
            scheduler_name = self.args.lr_schedule
            if scheduler_name == "constant" and self.args.lr_warmup > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=self.args.lr_warmup,
                num_training_steps=self.trainer.max_steps,
            )
            scheduler_config = {"frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }


def get_model_module(training_args):
    if training_args.model_name_or_path == 'tapex':
        model = transformers.BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")
        # potentially change model config
        # model.config.xxx = 'xxx'
        model_type_info = ModelTypeInfo(model_name_or_path=training_args.model_name_or_path,
                                        pad_token_id=1,
                                        mask_token_id=-100,
                                        input_targets=True,
                                        loss_out_id='loss',
                                        additional_inputs_mapping = {
                                            '*': None,
                                            'lm_labels': 'target',
                                            'label': 'target',
                                            }
                                        )
    else:
        # TODO try search path
        raise NotImplementedError(f"No initialization implemented for model {training_args.model_name_or_path}!")
    return LightningWrapper(model, training_args, model_type_info=model_type_info)


@dataclass
class ModelArgs:
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


class BasicLM(L.LightningModule):
    def __init__(
        self,
        training_args: "TrainingArgs",  # do in string to remove dependency when loading.
        adhoc_args: ModelArgs = ModelArgs(),
        effective_batch_size_per_step=-100_000,
        vocab_size=None,
        samples_processed=0.0,
        tokens_processed=0.0,
    ) -> None:
        super().__init__()
        if not training_args.resume_training:
            self.save_hyperparameters(
                ignore=["effective_batch_size_per_step", "samples_processed", "tokens_processed"]
            )
        self.args = training_args
        self.adhoc_args = adhoc_args
        config = AutoConfig.from_pretrained(self.args.model_name_or_path, return_dict=True)

        if self.args.language_modeling_strategy == "mlm":
            self.model: PreTrainedModel = (
                AutoModelForMaskedLM.from_pretrained(self.args.model_name_or_path, config=config)
                if not self.args.from_scratch
                else AutoModelForMaskedLM.from_config(config=config)
            )
        elif self.args.language_modeling_strategy == "clm":
            self.model: PreTrainedModel = (
                AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, config=config)
                if not self.args.from_scratch
                else AutoModelForCausalLM.from_config(config=config)
            )

        self.model.resize_token_embeddings(vocab_size)

        if self.args.from_scratch and get_rank() == 0:
            logger.info("Training from scratch without pretrained weights")

        self.effective_batch_size_per_step = effective_batch_size_per_step
        self.register_buffer("samples_processed", torch.tensor(samples_processed))
        self.register_buffer("tokens_processed", torch.tensor(tokens_processed))

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        self.samples_processed += self.effective_batch_size_per_step
        self.tokens_processed += self.effective_batch_size_per_step * self.args.max_sequence_length
        self.log_dict(
            {
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            rank_zero_only=True,
            on_step=True,
            on_epoch=False,
        )

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss

        self.log_dict(
            {
                "val/loss": loss,
                "progress/samples": self.samples_processed,
                "progress/tokens": self.tokens_processed,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        if self.global_rank == 0:
            logger.info(
                f"Using lr: {self.args.learning_rate}, weight decay: {self.args.weight_decay} and warmup steps: {self.args.lr_warmup}"
            )

        named_parameters = list(self.model.named_parameters())

        ### Filter out parameters that are not optimized (requires_grad == False)
        named_parameters = list(
            filter(lambda named_param: named_param[1].requires_grad, named_parameters)
        )

        ### Do not include LayerNorm and bias terms for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_parameters,
            self.args.learning_rate,
            betas=(self.adhoc_args.adam_beta1, self.adhoc_args.adam_beta2),
            eps=self.adhoc_args.adam_epsilon,
        )

        if self.args.lr_schedule == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, verbose=True
            )
            if self.args.lr_warmup > 0:  # Wrap ReduceLROnPlateau to enable LR warmup
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=self.args.lr_warmup,
                    after_scheduler=scheduler,
                )
            scheduler_config = {"frequency": self.args.val_frequency, "monitor": "train/loss"}
        else:
            scheduler_name = self.args.lr_schedule
            if scheduler_name == "constant" and self.args.lr_warmup > 0:
                scheduler_name += "_with_warmup"
            scheduler = get_scheduler(
                scheduler_name,
                optimizer,
                num_warmup_steps=self.args.lr_warmup,
                num_training_steps=self.trainer.max_steps,
            )
            scheduler_config = {"frequency": 1}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", **scheduler_config},
        }
