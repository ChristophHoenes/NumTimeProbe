import logging
import logging.config
import pickle
import traceback
from mock import Mock
from pathlib import PurePath
from typing import List, Type, Union

import torch
import wandb
from dargparser import dargparse
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, TapexTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM

import numerical_table_questions.data_synthesis
from numerical_table_questions.dlib.frameworks.wandb import WANDB_ENTITY, WANDB_PROJECT
from numerical_table_questions.arguments import TrainingArgs, MiscArgs, TokenizationArgs
from numerical_table_questions.data_loading import TableQADataModule
from numerical_table_questions.data_synthesis import Table, TableQuestionDataSet, QuestionTemplate, SQLColumnExpression, SQLOperator, SQLConditionTemplate, TableQuestion, execute_sql
from numerical_table_questions.metrics import token_accuracy
from numerical_table_questions.model import LightningWrapper
from numerical_table_questions.model_utils import get_model_module, get_model_specific_config


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TableQaModel:

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def predict(self,
                str_inputs: Union[str, List[str]],
                table_inputs: Union[Table, List[Table]],
                output_tokens: bool = False
                ) -> Union[str, torch.LongTensor]:
        # wrap input to lists
        if isinstance(str_inputs, str):
            str_inputs = [str_inputs]
        if isinstance(table_inputs, Table):
            table_inputs = [table_inputs]
        # TODO convert to tensors / dataframes
        tables = [table.pandas_dataframe for table in table_inputs]
        encoding = self._tokenizer(table=tables,
                                   query=str_inputs,
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt")
        outputs = self._model.generate(**encoding)
        if output_tokens:
            return outputs
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def tokenize(self, input: Union[str, Table], row_format: bool = True):
        if isinstance(input, Table):
            def linearize(data_dict):
                if row_format:
                    lin_data = [col for row in data_dict['rows'] for col in row]
                else:
                    lin_data = [row[col_idx]
                                for col_idx in range(data_dict['rows'][0])
                                for row in data_dict['rows']]
                return data_dict['header'] + lin_data

            linearized_string = linearize(input._data_dict)
        else:
            linearized_string = input
        return self.tokenizer(linearized_string)


class SemanticParsingModel(TableQaModel):

    def predict(self,
                str_inputs: Union[str, List[str]],
                table_inputs: Union[Table, List[Table]],
                output_tokens: bool = False
                ) -> Union[str, torch.LongTensor]:
        # wrap input in lists
        if not isinstance(str_inputs, list):
            str_inputs = [str_inputs]
        if not isinstance(table_inputs, list):
            table_inputs = [table_inputs]

        prompt_template = ("Given the following schema:\n"
                           "{name} ({schema})\n"
                           "Write a SQL query to answer the following question:\n"
                           "{question}")
        prompt = tuple([prompt_template.format(name=table_inputs[q].table_name,
                                               schema=table_inputs[q]._data_dict['header'],
                                               question=question)
                        for q, question in enumerate(str_inputs)])
        encoding = self._tokenizer(prompt, padding=True, truncation=True,
                                   return_tensors="pt")
        # TODO remove input_ids indexing?
        outputs = self._model.generate(encoding['input_ids'], max_new_tokens=512)
        if output_tokens:
            return outputs
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        query_predictions = [''.join(sample) for sample in decoded]
        answers = []
        for pred, sql in enumerate(query_predictions):
            try:
                result = execute_sql(sql, table_inputs[pred].pandas_dataframe)
            except Exception as e:
                logger.warn(f"Invalid answer encounterd (setting to None)! Reason: {e}")
                result = None
            answers.append(result)
        return answers, query_predictions


def evaluate(model: Type[TableQaModel],
             dataset: TableQuestionDataSet,
             metric: str = 'exact_match_accuracy'):
    tables = [question._table for question in dataset._questions]
    predictions = model.predict([q._nl_question for q in dataset.questions],
                                tables,
                                output_tokens=metric == 'token_accuracy'
                                )
    # Semantic parsing models return the originally predicted SQL query 
    # additionally to the answer via execution
    if isinstance(predictions, tuple):
        predictions, query_predictions = predictions

    ground_truth = dataset.ground_truth
    # TODO metric dict map name tu function then call function(**kwargs)
    if metric == 'exact_match_accuracy':
        metric_result, eval_results = exact_match_accuracy(predictions, ground_truth)
    elif metric == 'token_accuracy':
        metric_result, eval_results = token_accuracy(predictions, ground_truth)
    elif metric == 'fuzzy_match_accuracy':
        pass
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented!")
    return metric_result, predictions, ground_truth, eval_results


def evaluate_trained(eval_args, misc_args, tokenizer_args, model_checkpoint_path=None):
    resolved_checkpoint_path = model_checkpoint_path or eval_args.checkpoint_path
    # Initiallize Weights&Biases logger to log artefacts
    wandb_logger = WandbLogger(
        project=misc_args.wandb_project or WANDB_PROJECT,
        entity=WANDB_ENTITY,
        log_model="all",
        tags=['eval', *misc_args.wandb_tags],
        name=misc_args.wandb_run_name,
    )
    # Initialize trainer
    trainer = Trainer(
        max_steps=eval_args.training_goal,
        val_check_interval=eval_args.val_frequency,
        check_val_every_n_epoch=None,  # validation based on steps instead of epochs
        devices=eval_args.cuda_device_ids or eval_args.num_devices,
        accelerator=eval_args.accelerator,
        logger=wandb_logger,
        deterministic=True,
        precision=eval_args.precision,
        gradient_clip_val=eval_args.gradient_clipping,
        accumulate_grad_batches=eval_args.gradient_accumulation_steps,
        inference_mode=not eval_args.compile,  # inference_mode for val/test and PyTorch 2.0 compiler don't like each other  # noqa: E501
    )
    # for this to work the hyperparameters need to be saved and no positional arguments in LightningWrapper are allowed
    # alternatively overwrite load_from_checkpoint but not recommended since some other side effects from super() might be lost and model is still needed
    # model = LightningWrapper.load_from_checkpoint(checkpoint_path=resolved_checkpoint_path)
    model_module = get_model_module(eval_args.model_name_or_path)
    model_kwargs = get_model_specific_config(eval_args.model_name_or_path)
    model = LightningWrapper(model_module, eval_args, **model_kwargs)
    # TODO check local first then wandb
    if resolved_checkpoint_path:
        checkpoint_content = torch.load(resolved_checkpoint_path)
        model.load_state_dict(checkpoint_content["state_dict"])
    dm = TableQADataModule(model.model_specs,
                           table_corpus=eval_args.table_corpus,
                           dataset_name=eval_args.dataset_name,
                           train_batch_size=eval_args.batch_size_per_device,
                           eval_batch_size=eval_args.eval_batch_size_per_device,
                           lazy_data_processing=args.lazy_data_processing,
                           is_batch_dict=args.is_batch_dict,
                           data_dir=args.data_dir,
                           tokenizing_args=tokenizer_args,
                           num_dataloader_workers=eval_args.workers,
                           too_many_open_files_fix=misc_args.too_many_open_files_fix,
                           )
    trainer.test(model, datamodule=dm)


def main(model_name, dataset_version, **kwargs):
    if model_name == 'tapex':
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
        model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")
        tqa_model = TableQaModel(model, tokenizer)
    if dataset_version == 'basic_dataset':
        dataset = data_synthesis.caching('./data/NumTabQA/.cache', 'basic_dataset.pickle')  # invalid api
    else:
        raise FileNotFoundError(f"No saved file for dataset version '{dataset_version}' was found!")
    metric_result, predictions, ground_truth, eval_results = evaluate(tqa_model, dataset)
    """ TODO connect to wandb runs with evaluation artifacts
    evaluation_dict = {'model_name': model_name,
                       'run_details': dict(),  # TODO fill with content
                       'metric_result': metric_result,
                       'predictions': predictions,
                       'ground_truth': ground_truth,
                       'eval_results': eval_results
                       }
    pickle.dump(evaluation_dict)
    """
    print(metric_result)
    return metric_result


if __name__ == "__main__":
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args, misc_args, tokenizer_args = dargparse(dataclasses=(TrainingArgs, MiscArgs, TokenizationArgs))
    run = wandb.init(project="table-qa-debug", job_type="add-log")
    try:
        # old training data (agg count 1 a lot)
        #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/7425bh3s/checkpoints/snap-1040-samples-199616.0-204406784.0-loss-0.05.ckpt')
        # new training data (agg count 20%)
        #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/2kix9c4k/checkpoints/last_model_ckpt.ckpt')
        # new training data (agg count 0%)
        #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/0keck68y/checkpoints/last_model_ckpt.ckpt')
        # zero shot
        #evaluate_trained(args, misc_args, tokenizer_args)
        # trained with lazy processing (e.g. truncating too long tables)
        evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/v6o1yucb/checkpoints/last_model_ckpt.ckpt')
        wandb.finish()
    except:
        logger.error("Uncaught exception: %s", traceback.format_exc())
        raise SystemExit
    finally:
        artifact = wandb.Artifact("run.log", type="logfile")
        artifact.add_file("../../run.log")
        wandb.log_artifact(artifact)
        wandb.finish()
