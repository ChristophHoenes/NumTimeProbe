import os
import logging
import logging.config
from datetime import datetime
from pathlib import PurePath
from typing import List, Type, Union

import datasets
import torch
from accelerate import PartialState
from dargparser import dargparse
from transformers import TapexTokenizer, BartForConditionalGeneration

from numerical_table_questions.arguments import TrainingArgs, MiscArgs, TokenizationArgs, DataProcessingArgs
from numerical_table_questions.data_loading import TableQADataModule, SQLCoderDataModule, create_sqlcoder_dataset
from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.utils.lightning_utils import get_lightning_trainer
from numerical_table_questions.metrics import token_accuracy, str_match_accuracy
from numerical_table_questions.model import LightningWrapper, SQLCoder
from numerical_table_questions.sqlcoder_model import (
    sqlcoder_generation,
    get_sqlcoder_inference_pipeline
    )
from numerical_table_questions.utils.data_utils import caching
from numerical_table_questions.utils.model_utils import get_model_module, get_model_specific_config, extract_model_name
from numerical_table_questions.utils.sql_utils import execute_sql
from numerical_table_questions.utils.tokenizer_utils import get_tokenizer
from numerical_table_questions.utils.system_helpers import parse_auto_arguments
from numerical_table_questions.utils.wandb_utils import try_load_local_then_wandb_checkpoint, get_wandb_logger


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


def evaluate_trained(eval_args, misc_args, tokenizer_args, data_args=DataProcessingArgs(), model_checkpoint_path=None):
    resolved_checkpoint_path = model_checkpoint_path or eval_args.checkpoint_path
    model_name = extract_model_name(eval_args.model_name_or_path)
    # Initiallize Weights&Biases logger to log artefacts
    tags = ['eval', model_name]
    wandb_logger = get_wandb_logger(misc_args, tags=tags)
    parse_auto_arguments(eval_args)
    # Initialize trainer
    trainer = get_lightning_trainer(eval_args, wandb_logger=wandb_logger)

    if model_name == 'sqlcoder':
        if eval_args.optimize_inference_pipeline:
            evaluate_sqlcoder(eval_args, misc_args, tokenizer_args, data_args, wandb_logger)
            return
        else:
            model = SQLCoder(eval_args.model_name_or_path)
    else:
        # for this to work the hyperparameters need to be saved and no positional arguments in LightningWrapper are allowed
        # alternatively overwrite load_from_checkpoint but not recommended since some other side effects from super() might be lost and model is still needed
        # model = LightningWrapper.load_from_checkpoint(checkpoint_path=resolved_checkpoint_path)
        model_module = get_model_module(eval_args.model_name_or_path)
        model_kwargs = get_model_specific_config(eval_args.model_name_or_path)
        model = LightningWrapper(model_module, eval_args, **model_kwargs)

    # try load a checkpoint if provided
    if resolved_checkpoint_path:
        try_load_local_then_wandb_checkpoint(model, wandb_logger, resolved_checkpoint_path, map_location=eval_args.accelerator)

    if model_name == 'sqlcoder':
        data_module_class = SQLCoderDataModule
    else:
        data_module_class = TableQADataModule
    # Initialize the data module that is appropriate for the model
    dm = data_module_class(model.model_specs,
                           table_corpus=eval_args.table_corpus_name,
                           dataset_name=eval_args.dataset_suffix,
                           train_batch_size=eval_args.batch_size_per_device,
                           eval_batch_size=eval_args.eval_batch_size_per_device,
                           lazy_data_processing=eval_args.lazy_data_processing,
                           is_batch_dict=eval_args.is_batch_dict,
                           data_dir=eval_args.data_dir,
                           tokenizing_args=tokenizer_args,
                           num_dataloader_workers=eval_args.workers,
                           too_many_open_files_fix=misc_args.too_many_open_files_fix,
                           )
    trainer.test(model, datamodule=dm)


def evaluate_sqlcoder(eval_args=TrainingArgs(), misc_args=MiscArgs(), tokenizer_args=TokenizationArgs(), data_args=DataProcessingArgs(), wandb_logger=None):
    parse_auto_arguments(eval_args)  # maybe not needed because accelerate determines gpus
    post_processing_fn = lambda x: [item.strip() for item in x]
    metric_function = str_match_accuracy
    if wandb_logger is None:
        wandb_logger = get_wandb_logger(misc_args, tags=['sqlcoder', 'eval'])

    created_datasets = {}
    for split in data_args.splits:
        cache_file_name = f"sqlcoder_{data_args.table_corpus}_{data_args.dataset_name or '-'.join(data_args.template_names)}_{split}"
        save_path = os.path.join(data_args.cache_dir,
                                 cache_file_name
                                 )
        if os.path.exists(save_path):
            created_datasets[split] = caching(cache_file_name=cache_file_name, cache_path=data_args.cache_dir)
        else:
            created_datasets[split] = create_sqlcoder_dataset(eval_args, tokenizer_args, split=split, save_path=save_path)

    model = get_model_module(eval_args.model_name_or_path)
    tokenizer = get_tokenizer(eval_args.model_name_or_path)

    distributed_state = PartialState()
    pipe = get_sqlcoder_inference_pipeline(model, tokenizer)
    step_size = len(created_datasets['test']) // distributed_state.num_processes
    data_shards = [
        created_datasets['test'].select(range(
            i * step_size,
            # last shard includes all remaining samples
            (i + 1) * step_size if i < (distributed_state.num_processes-1) else len(created_datasets['test'])
            ))
        for i in range(distributed_state.num_processes)
        ]
    string_predictions = []
    with distributed_state.split_between_processes(data_shards) as dataset_shard:
        if isinstance(dataset_shard, list) and len(dataset_shard) > 1:
            dataset_shard = datasets.concatenate_datasets(dataset_shard)
        else:
            dataset_shard = dataset_shard[0]
        if not isinstance(dataset_shard, datasets.Dataset):
            raise TypeError(f"Expected dataset shard to be of type datasets.Dataset but encountered {type(dataset_shard)}!")
        string_predictions.extend(sqlcoder_generation(dataset_shard, pipe=pipe, batch_size=eval_args.eval_batch_size_per_device))

    # calculate metric
    processed_predictions = post_processing_fn(string_predictions if isinstance(string_predictions[0], str) else [pred[0] for pred in string_predictions])
    processed_targets = post_processing_fn(created_datasets['test']['answers'])
    metric_outputs = metric_function(processed_predictions, processed_targets)
    # log metric result
    # only consider first returned value if metric has multiple return values, which is main result by convention (others are supplemental information)
    metric_results = {f"test/{metric_function.__name__}": metric_outputs[0] if isinstance(metric_outputs, tuple) else metric_outputs}

    print(metric_results)
    if wandb_logger is not None:
        wandb_logger.log_metrics(metric_results)

        predictions = [[pred, processed_predictions[i], sql] for i, pred, sql in enumerate(string_predictions)]
        wandb_logger.log_text(key='predictions', columns=['text_predictions', 'post_processed_answer', 'sql'], data=predictions)
    else:
        predictions_save_path = os.path.join(
            data_args.cache_dir,
            f"sqlcoder_{data_args.table_corpus}_{data_args.dataset_name or '-'.join(data_args.template_names)}_test_predictions.txt"
            )
        with open(predictions_save_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}\n")
            f.write(f"Metric results: {metric_results}\n")
            formated_predictions = [f"SQL: {sql}\nText Answer: {pred}\nPost-processed Answer: {processed_predictions[i]}" for i, pred, sql in enumerate(string_predictions)]
            prediction_lines = '\n'.join(formated_predictions)
            f.write(f"Predictions:\n{prediction_lines}\n")


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
    #import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args, misc_args, tokenizer_args, data_args = dargparse(dataclasses=(TrainingArgs, MiscArgs, TokenizationArgs, DataProcessingArgs))
    # old training data (agg count 1 a lot)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/7425bh3s/checkpoints/snap-1040-samples-199616.0-204406784.0-loss-0.05.ckpt')
    # new training data (agg count 20%)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/2kix9c4k/checkpoints/last_model_ckpt.ckpt')
    # new training data (agg count 0%)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/0keck68y/checkpoints/last_model_ckpt.ckpt')
    # zero shot
    evaluate_trained(args, misc_args, tokenizer_args, data_args)
    # trained with lazy processing (e.g. truncating too long tables)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/v6o1yucb/checkpoints/last_model_ckpt.ckpt')
    # trained lazy diff
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/0w076ku1/checkpoints/last_model_ckpt.ckpt')
    # fine-tuned TAPAS
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/9n4lmvw1/checkpoints/last_model_ckpt.ckpt')
    # fine-tuned Tapex all standard templates larger batch size (gas)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/sjwf0hff/checkpoints/last_model_ckpt.ckpt')
    # fine-tuned Tapex all standard templates smaller batch size (gas)
    #evaluate_trained(args, misc_args, tokenizer_args, 'table-qa-debug/ywup1ksg/checkpoints/last_model_ckpt.ckpt')
    # fine-tuned OmniTab
    #evaluate_trained(args, misc_args, tokenizer_args, model_checkpoint_path='table-qa-debug/izc1diit/checkpoints/last_model_ckpt.ckpt')
