import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union, Dict, Callable, Tuple, List

import lightning as L
import torch
from loguru import logger
from torch.optim import AdamW
from torchmetrics import MetricCollection
from transformers.optimization import get_scheduler
from warmup_scheduler import GradualWarmupScheduler

from numerical_table_questions.model_utils import ModelTypeInfo, model_specific_generation
from numerical_table_questions.tokenizer_utils import get_tokenizer, convert_to_long_tensor_if_int_tensor

if TYPE_CHECKING:
    from train import TrainingArgs


@dataclass
class OptimizerArgs:
    learning_rate: float
    optimizer_class: torch.optim.Optimizer = AdamW
    kwargs: dict = field(
        default_factory=lambda: {
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    )

    def __post_init__(self):
        # pass learning rate as kwarg to the optimizer_class
        self.kwargs['lr'] = self.learning_rate


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
                 loss_fn: Callable = torch.nn.functional.nll_loss,
                 # TODO make metrics a metrics collection and call/log automatically with lightning
                 # (see docs https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html)
                 metrics: MetricCollection = MetricCollection([]),
                 generation_metrics: Dict[str, Union[Callable, Tuple[Callable, dict]]] = dict(),
                 optimizer_args: OptimizerArgs = None,
                 effective_batch_size_per_step=None,
                 # keep track of how much data the model has seen, must be float for easier logging
                 samples_processed: float = 0.0,
                 tokens_processed: float = 0.0,
                 ):
        _nn_module_arguments = [name for name, value in locals().items() if isinstance(value, torch.nn.Module)]
        super().__init__()
        if not training_args.resume_training:
            self.save_hyperparameters(
                ignore=(
                    ["effective_batch_size_per_step",
                     # no need to save input arguments as state changes during training and
                     # they are registered as buffers which are saved with the model checkpoint
                     "samples_processed", "tokens_processed"]
                    + _nn_module_arguments
                    )
                )  # nn.Module objects are already saved within a checkpoint file
        self.model = model
        self.tokenizer = None
        self.model_specs = model_type_info or ModelTypeInfo(training_args.model_name_or_path)
        self.loss_fn = loss_fn
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        # generation metrics use the outputs of the generate method rather than the ones of the forward pass
        self.generation_metrics = {
            metric_name: tup[0] if isinstance(tup, tuple) else tup  # syntax sugar, no need to provide tuple if no config is passed
            for metric_name, tup in generation_metrics.items()
            }  # extract name2callable mapping for metric function
        self.generation_metrics_config = {
            metric_name: tup[1] for metric_name, tup in generation_metrics.items()
            if isinstance(tup, tuple)
            }  # extract name2config mapping for metric function
        # generation requires tokenizer for decoding the IDs
        if len(self.generation_metrics) > 0:
            self.tokenizer = get_tokenizer(self.model_specs.model_name_or_path)
        # prepare metric requirements according to provided config
        for gen_metric in self.generation_metrics:
            self._get_metric_requirements(gen_metric)
        self.optimizer_args = optimizer_args or OptimizerArgs(learning_rate=training_args.learning_rate)
        self.args = training_args
        if effective_batch_size_per_step is None:
            warnings.warn("No effective batch size was set! Using negative batch_size_per_device as proxy."
                          "This will lead to inacurate values for 'samples_processed' and 'tokens_processed'.")
            self.effective_batch_size_per_step = -training_args.batch_size_per_device
        else:
            self.effective_batch_size_per_step = effective_batch_size_per_step
        # when resuming from a checkpoint samples_processed and tokens_processed should already exist as buffers
        # for a new (fresh created) model register as buffers with provided start values (most likely 0)
        if not training_args.resume_training:
            self.register_buffer("samples_processed", torch.tensor(samples_processed))
            self.register_buffer("tokens_processed", torch.tensor(tokens_processed))
        self.predictions = []

    # TODO make custom metric class and each metric defines its own requirements -> get rid of dependency with data_loading
    def _get_metric_requirements(self, metric_name):
        match metric_name:
            case 'exaple_custom_metric_preparation':
                # str_match_accuracy requires tokenizer
                # self.tokenizer = get_tokenizer(self.model_specs.model_name_or_path)
                # self.generation_metrics_requirements['kwargs']['tokenizer'] = self.tokenizer  # <- not needed currently
                pass
            case _:
                warnings.warn(f"Unknown generation_metric {metric_name}! No specific preparation is executed.")

    def forward(self, inputs, target=None):
        if self.model_specs.input_targets and target is None:
            raise ValueError("Configuration argument 'input_targets' is set to True but no targets were provided as argument! "
                             "Please call the model forward pass with inputs and tragets.")
        if isinstance(inputs, dict):
            forward_kwargs = {model_input_id: inputs[data_input_id] if data_input_id != 'targets' else target
                              for model_input_id, data_input_id in self.model_specs.dict_input_mapping.items()
                              if isinstance(model_input_id, str)
                              }
            # make sure all int tensors to have dtype long; non-int-tensors stay unchanged
            forward_kwargs = {key: convert_to_long_tensor_if_int_tensor(value)
                              for key, value in forward_kwargs.items()
                              }
            outputs = self.model(**forward_kwargs)
        else:
            # always wrap inputs in tuple
            if isinstance(inputs, list):
                inputs = tuple(inputs)
            elif not isinstance(inputs, tuple):
                inputs = (inputs,)
            # use input order specified in model_specs
            if len(self.model_specs.tensor_input_mapping) > 0:
                # compute input format specified via self.model_specs.input_mapping
                forward_args = order_positional_arguments(inputs, target, input_map=self.model_specs.tensor_input_mapping)
                forward_kwargs = {model_input_id: data_function(inputs, target)
                                  for model_input_id, data_function in self.model_specs.tensor_input_mapping.items()
                                  if isinstance(model_input_id, str)
                                  }
                # make sure all int tensors to have dtype long; non-int-tensors stay unchanged
                forward_args = [convert_to_long_tensor_if_int_tensor(item) for item in forward_args]
                forward_kwargs = {key: convert_to_long_tensor_if_int_tensor(value)
                                  for key, value in forward_kwargs.items()
                                  }
                outputs = self.model(*forward_args, **forward_kwargs)
            elif self.model_specs.input_targets:
                outputs = self.model(*inputs, target)
            else:
                outputs = self.model(*inputs)
        # always wrap outputs in dict
        if isinstance(outputs, (list, tuple)):
            outputs = {i: output for i, output in enumerate(outputs)}
        elif not isinstance(outputs, dict):
            outputs = {0: outputs}
        return outputs

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            target = batch.get('targets')
            inputs = {key: value for key, value in batch.items() if key != 'targets'}
        else:
            inputs, target = batch

        # only consider first couple of input tokens as NL-question-only sanity check
        if self.args.only_first_x_tokens > 0:
            if isinstance(batch, dict):
                inputs['input_ids'][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs['attention_mask'][:, self.args.only_first_x_tokens:] = 0
            else:
                inputs[0][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs[1][:, self.args.only_first_x_tokens:] = 0

        outputs = self(inputs, target)

        if self.model_specs.loss_out_id is None:
            # compute loss
            predictions = outputs[self.model_specs.prediction_scores_out_id]
            loss = self.loss_fn(predictions, target.view_as(predictions))
        else:
            # retrieve loss
            loss = outputs[self.model_specs.loss_out_id]
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False)

        # compute and log torchmetrics at every step
        torchmetrics_out = self.train_metrics(outputs, target)
        self.log_dict(torchmetrics_out)
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
        if isinstance(batch, dict):
            target = batch.get('targets')
            inputs = {key: value for key, value in batch.items() if key != 'targets'}
        else:
            inputs, target = batch
        # only consider first couple of input tokens as NL question only sanity check
        # TODO improve by saving question end ids in data processing? probably too much effort an little reward
        if self.args.only_first_x_tokens > 0:
            if isinstance(batch, dict):
                inputs['input_ids'][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs['attention_mask'][:, self.args.only_first_x_tokens:] = 0
            else:
                inputs[0][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs[1][:, self.args.only_first_x_tokens:] = 0
        outputs = self(inputs, target)
        if self.model_specs.loss_out_id is None:
            # compute loss
            loss = self.loss_fn(outputs[self.model_specs.prediction_scores_out_id], target.view(-1))
        else:
            # retrieve loss
            loss = outputs[self.model_specs.loss_out_id]

        # moving everything that is logged to GPU to workaround this issue https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        self.log_dict(
            {
                "val/loss": loss.item(),
                "progress/samples": self.samples_processed.to(self.device),
                "progress/tokens": self.tokens_processed.to(self.device),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # update state of torchmetrics
        self.valid_metrics.update(outputs, target)

    def on_validation_epoch_end(self):
        # compute and log torchmetrics final epoch results
        output = self.valid_metrics.compute()
        # moving everything that is logged to GPU to workaround this issue https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        self.log_dict({k: v.to(self.device) for k, v in output})
        self.valid_metrics.reset()

    def test_step(self, batch, batch_idx):
        # TODO log model predictions token and text
        if isinstance(batch, dict):
            target = batch.get('targets')
            inputs = {key: value for key, value in batch.items() if key != 'targets'}
        else:
            inputs, target = batch
        # only consider first couple of input tokens as NL question only sanity check
        # TODO improve by saving question end ids in data processing? probably too much effort an little reward
        if self.args.only_first_x_tokens > 0:
            if isinstance(batch, dict):
                inputs['input_ids'][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs['attention_mask'][:, self.args.only_first_x_tokens:] = 0
            else:
                inputs[0][:, self.args.only_first_x_tokens:] = self.model_specs.pad_token_id
                inputs[1][:, self.args.only_first_x_tokens:] = 0

        # only compute normal forward step if test_loss is explicitly required or at least one torchmetric uses its outputs
        if self.args.do_forward_in_test_step or self.args.force_test_loss_computation or len(self.test_metrics) > 0:
            outputs = self(inputs, target)

            if len(self.test_metrics) > 0:
                # update state of torchmetrics
                self.test_metrics.update(outputs, target)

            if self.args.force_test_loss_computation:
                if self.model_specs.loss_out_id is None:
                    # compute loss
                    loss = self.loss_fn(outputs[self.model_specs.prediction_scores_out_id], target.view(-1))
                else:
                    # retrieve loss
                    loss = outputs[self.model_specs.loss_out_id]

                # moving everything that is logged to GPU to workaround this issue https://github.com/Lightning-AI/pytorch-lightning/issues/18803
                self.log_dict(
                    {
                        "test/loss": loss.item(),
                        # moving everything that is logged to GPU to workaround this issue https://github.com/Lightning-AI/pytorch-lightning/issues/18803
                        # this should not be necessary because buffers should automatically be on the same device...
                        # "progress/samples": self.samples_processed.to(self.device),
                        # "progress/tokens": self.tokens_processed.to(self.device),
                        "progress/samples": self.samples_processed.item(),
                        "progress/tokens": self.tokens_processed.item(),
                    },
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        else:
            outputs = None

        # if at least one metric that evaluates generation results exists execute generation (can be more expensive than a simple forward pass)
        if len(self.generation_metrics) > 0:
            if isinstance(batch, dict):
                text_targets = batch['answers']
                input_ids = inputs['input_ids']
            else:
                input_ids = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
                try:
                    # if targets are not provided as text, create text_targets by decoding the ids
                    if not isinstance(target, tuple):
                        safe_target = torch.where(target != self.model_specs.mask_token_id, target, self.model_specs.pad_token_id)
                        text_targets = self.tokenizer.batch_decode(safe_target, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    else:
                        text_targets = target[1]  # if tuple assume second position is text targets
                except KeyError as e:
                    raise ValueError("Batch must be provided as a dictionary "
                                     "since the TensorDataset (in-memory) configuration does not support string items!"
                                     ) from e
            # TODO test if this leaks to much information (e.g are results worse if this is hard coded to 20)
            max_target_len = max([len(sample) for sample in text_targets])
            test_dataset = self.trainer.test_dataloaders.dataset
            string_predictions = model_specific_generation(self.args.model_name_or_path, self.model, self.tokenizer, inputs, outputs, max_target_len=max_target_len, test_dataset=test_dataset)
            self.predictions.extend(string_predictions)

            # apply post processing specified for metric to both, generated prediction and text targets
            batch_metric_results = dict()
            for metric_name, metric_function in self.generation_metrics.items():
                # TODO post processing
                config = self.generation_metrics_config.get(metric_name)
                if config is not None:
                    post_processing_fn = config.get('post_processing_fn')
                    metric_kwargs = config.get('kwargs', dict())
                    if post_processing_fn is not None:
                        processed_predictions = post_processing_fn(string_predictions)
                        processed_targets = post_processing_fn(text_targets)
                    else:  # no post processing
                        processed_predictions = string_predictions
                        processed_targets = text_targets
                else:  # no post processing
                    processed_predictions = string_predictions
                    processed_targets = text_targets
                    metric_kwargs = dict()
                # compute generation metric on postprocessed texts
                metric_outputs = metric_function(processed_predictions, processed_targets, **metric_kwargs)
                # log metric result
                # only consider first returned value if metric has multiple return values, which is main result by convention (others are supplemental information)
                batch_metric_results[f"test/{metric_name}"] = metric_outputs[0] if isinstance(metric_outputs, tuple) else metric_outputs
                # TODO handling of logging additional outputs, if neccesary for some metric

            self.log_dict(
                batch_metric_results,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                )

    def on_test_epoch_end(self):
        # compute and log torchmetrics final epoch results
        output = self.test_metrics.compute()
        # moving everything that is logged to GPU to workaround this issue https://github.com/Lightning-AI/pytorch-lightning/issues/18803
        self.log_dict({k: v.to(self.device) for k, v in output})
        self.test_metrics.reset()
        # log and clear predictions
        text_predictions = [[pred] for pred in self.predictions]
        # table = wandb.Table(data=text_predictions, columns=['text_predictions'])
        self.logger.log_table(key='text_predictions', columns=['text_predictions'], data=text_predictions)  # assumes wandb logger
        self.predictions = []

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
        optimizer = self.optimizer_args.optimizer_class(optimizer_parameters,
                                                        **self.optimizer_args.kwargs)
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
