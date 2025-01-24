import os
from datetime import datetime
from typing import List, Optional, Union

import datasets
import lightning as L
import torch
from accelerate import PartialState
from dargparser import dargparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from numerical_table_questions.arguments import TrainingArgs, MiscArgs, TokenizationArgs, DataProcessingArgs
from numerical_table_questions.evaluation import get_wandb_logger, parse_auto_arguments  # move to diferent utils module
from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_loading import SQLCoderDataModule
from numerical_table_questions.metrics import str_match_accuracy
from numerical_table_questions.model_utils import ModelTypeInfo
from numerical_table_questions.sql_utils import execute_sql


PROMPT_TEMPLATE = ("### Task\n"
                   "Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]\n\n"
                   "### Database Schema\n"
                   "The query will run on a database with the following schema:\n"
                   "{table_metadata_string_DDL_statements}\n\n"
                   "### Answer\n"
                   "Given the database schema, here is the SQL query that [QUESTION]{user_question}[/QUESTION]\n"
                   "[SQL]"
                   )


def sqlcoder_prompt_template(user_question: Optional[str] = None, table: Optional[dict] = None) -> str:
    if user_question is None:
        user_question = "{user_question}"
    if table is None:
        table_ddl = "{table_metadata_string_DDL_statements}"
    else:
        table_ddl_template = ("CREATE TABLE {table_name} (\n"
                              "{column_definitions}\n"
                              ");\n"
                              )
        table_name = table['table_name']
        column_definitions = [f"{column} {'FLOAT' if table['inferred_column_types'][c].lower() == 'numeric' else 'TEXT'},"
                              for c, column in enumerate(table['data_dict']['header'])
                              ]
        if len(column_definitions) > 0:
            column_definitions[-1] = column_definitions[-1][:-1]  # remove trailing comma of last row
        column_definitions = '\n'.join(column_definitions)
        table_ddl = table_ddl_template.format(table_name=table_name, column_definitions=column_definitions)

    return PROMPT_TEMPLATE.format(user_question=user_question, table_metadata_string_DDL_statements=table_ddl)


def get_sqlcoder_tokenizer(model_path: str = "defog/sqlcoder-7b-2", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
    return tokenizer


def get_sqlcoder_model(model_path: str = "defog/sqlcoder-7b-2"):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )


def get_sqlcoder_inference_pipeline(model=None, tokenizer=None, max_query_length: int = 300, num_beams: int = 5):
    if model is None:
        model = get_sqlcoder_model()
    if tokenizer is None:
        tokenizer = get_sqlcoder_tokenizer()
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_query_length,
        do_sample=False,
        return_full_text=False,  # added return_full_text parameter to prevent splitting issues with prompt
        num_beams=num_beams,  # recommended to do beam search with 5 beams for high quality results
        device_map="auto",
    )


def run_pipeline(pipe, prompts: Union[List[str], datasets.Dataset], batch_size: int = 8) -> List[str]:
    # make sure the model stops generating at triple ticks
    # eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]
    eos_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.pad_token_id = eos_token_id
    pipe.tokenizer.pad_token = '<pad>'

    generated_queries = pipe(
            prompts,
            batch_size=batch_size,
            num_return_sequences=1,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id
        )
    # postprocess generated queries
    generated_queries = [
        generated_query[0]['generated_text']
        .split(';')[0]
        .split('```')[0]
        .strip()
        + ';'
        for generated_query in generated_queries
    ]
    return generated_queries


def run_inference(question: Union[str, List[str]], table: Union[dict, List[dict]], model=None, tokenizer=None, pipe=None) -> Union[str, List[str]]:
    is_single_query = False
    if pipe is None:
        pipe = get_sqlcoder_inference_pipeline(model, tokenizer)
    if isinstance(question, str):
        question = [question]
        is_single_query = True

    prepared_prompts = []
    for i, q in enumerate(question):
        prompt = sqlcoder_prompt_template(q, table[i] if isinstance(table, list) else table)
        if '{' in prompt:
            raise ValueError(f"Prompt contains unresolved template marker:\n{prompt}")
        prepared_prompts.append(prompt)

    generated_queries = run_pipeline(pipe,
                                     prompts=prepared_prompts if len(prepared_prompts) > 1 else prepared_prompts[0],
                                     batch_size=len(prepared_prompts)
                                     )

    if is_single_query:
        return generated_queries[0]
    return generated_queries


def sqlcoder_generation(question: Union[str, List[str]], table: Optional[Union[dict, List[dict]]] = None, model=None, tokenizer=None, pipe=None) -> Union[str, List[str]]:
    is_single_query = False
    if pipe is None:
        pipe = get_sqlcoder_inference_pipeline(model, tokenizer)

    if isinstance(question, datasets.Dataset):
        generated_query = run_pipeline(pipe, question)
    else:
        table_length = 'None' if table is None else len(table) if isinstance(table, list) else 1
        question_length = len(question) if isinstance(question, list) else 1
        if str(table_length) != str(question_length):
            raise ValueError(f"Table must be provided for every question but question has length {question_length} and table {table_length}!")
        generated_query = run_inference(question, table, pipe)

    if isinstance(generated_query, str):
        generated_query = [generated_query]
        is_single_query = True
    string_results = []
    for q, query in enumerate(generated_query):
        tab = table[q] if isinstance(table, list) else table
        tab = Table.from_state_dict(tab)
        # execute_sql always uses df as table name
        query = query.replace(tab.table_name, 'df')
        query_result = execute_sql(query, tab.pandas_dataframe)
        # if error or no rows return empty string else retrieve first cell from answer (single-number answers expected)
        if query_result is not None and len(query_result) > 0:  # at least one row
            string_results.append(str(query_result.iloc[0, 0]))
        else:
            string_results.append('')
    if is_single_query:
        return string_results[0]
    return string_results


class SQLCoder(L.LightningModule):
    def __init__(self, model_name_or_path: str, post_processing_fn=lambda x: [item.strip() for item in x], metric_function=str_match_accuracy, metric_name='str_match_accuracy', model_type_info=None):
        super().__init__()
        self.model = get_sqlcoder_model() if model_name_or_path.lower() == 'sqlcoder' else get_sqlcoder_model(model_name_or_path)
        self.tokenizer = get_sqlcoder_tokenizer() if model_name_or_path.lower() == 'sqlcoder' else get_sqlcoder_tokenizer(model_name_or_path)
        self.pipeline = get_sqlcoder_inference_pipeline(self.model, self.tokenizer)
        self.model_specs = model_type_info or ModelTypeInfo(model_name_or_path)
        self.predictions = []
        self.post_processing_fn = post_processing_fn
        self.metric_function = metric_function
        self.metric_name = metric_name

    def test_step(self, batch, batch_idx):
        # generate predictions for batch contents
        questions = batch['questions']
        tables = batch['tables']
        text_targets = batch['answers']
        string_predictions = sqlcoder_generation(questions, tables, self.model, self.tokenizer, self.pipeline)
        self.predictions.extend(string_predictions)
        # calculate metric
        processed_predictions = self.post_processing_fn(string_predictions)
        processed_targets = self.post_processing_fn(text_targets)
        metric_outputs = self.metric_function(processed_predictions, processed_targets)
        # log metric result
        # only consider first returned value if metric has multiple return values, which is main result by convention (others are supplemental information)
        batch_metric_results = {f"test/{self.metric_name}": metric_outputs[0] if isinstance(metric_outputs, tuple) else metric_outputs}
        self.log_dict(
                batch_metric_results,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch['questions']),
                )

    def on_test_epoch_end(self):
        # log and clear predictions
        # TODO check if this is on GPU and could lead to memory leak (e.g. is released after logging?)
        text_predictions = [[pred] for pred in self.predictions]
        # table = wandb.Table(data=text_predictions, columns=['text_predictions'])
        self.logger.log_table(key='text_predictions', columns=['text_predictions'], data=text_predictions)  # assumes wandb logger
        self.predictions.clear()


def create_sqlcoder_dataset(eval_args: Optional[TrainingArgs] = None,
                            tokenizer_args: Optional[TokenizationArgs] = None,
                            split: str = 'test',
                            model_type_info: Optional[ModelTypeInfo] = None,
                            save_path: Optional[str] = None
                            ) -> datasets.Dataset:
    # if no arguments were provided use default values
    if eval_args is None:
        eval_args = TrainingArgs()
    if tokenizer_args is None:
        tokenizer_args = TokenizationArgs()

    # Initialize the data module that is appropriate for the model
    dm = SQLCoderDataModule(model_type_info or ModelTypeInfo(eval_args.model_name_or_path),
                            table_corpus=eval_args.table_corpus,
                            dataset_name=eval_args.dataset_name,
                            train_batch_size=eval_args.batch_size_per_device,
                            eval_batch_size=eval_args.eval_batch_size_per_device,
                            lazy_data_processing=eval_args.lazy_data_processing,
                            is_batch_dict=eval_args.is_batch_dict,
                            data_dir=eval_args.data_dir,
                            tokenizing_args=tokenizer_args,
                            num_dataloader_workers=eval_args.workers,
                            )
    match split.lower():
        case 'test':
            dm.setup('test')
            dataloader = dm.test_dataloader()
        case 'train':
            dm.setup('fit')
            dataloader = dm.train_dataloader()
        case 'validation':
            dm.setup('validate')
            dataloader = dm.val_dataloader()
        case _:  # default to test
            raise ValueError(f"Invalid split: {split}")

    batch_datasets = []
    for batch in dataloader:
        batch_questions = []
        for sample in len(batch['questions']):
            batch_questions.append(sqlcoder_prompt_template(batch['questions'][sample], batch['tables'][sample]))
        batch['questions'] = batch_questions
        batch_datasets.append(datasets.Dataset.from_dict(batch))
    dataset = datasets.concatenate_datasets(batch_datasets)
    if save_path is not None:
        save_version(dataset, save_path)
    return dataset


if __name__ == "__main__":
    args, misc_args, tokenizer_args, data_args = dargparse(dataclasses=(TrainingArgs, MiscArgs, TokenizationArgs, DataProcessingArgs))
    parse_auto_arguments(args)  # maybe not needed because accelerate determines gpus
    post_processing_fn = lambda x: [item.strip() for item in x]
    metric_function = str_match_accuracy
    wandb_logger = get_wandb_logger(misc_args, tags=['sqlcoder', 'eval'])

    created_datasets = {}
    for split in data_args.splits:
        cache_file_name = f"sqlcoder_{data_args.table_corpus}_{data_args.dataset_name or '-'.join(data_args.template_names)}_{split}"
        save_path = os.path.join(data_args.data_dir,
                                 cache_file_name
                                 )
        if os.path.exists(save_path):
            created_datasets[split] = caching(cache_file_name=cache_file_name, cache_path=data_args.data_dir)
        else:
            created_datasets[split] = create_sqlcoder_dataset(args, tokenizer_args, split=split, save_path=save_path)
    raise
    model = get_sqlcoder_model(args.model_name_or_path)
    tokenizer = get_sqlcoder_tokenizer(args.model_name_or_path)
    pipe = get_sqlcoder_inference_pipeline(model, tokenizer)

    distributed_state = PartialState()
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
    pipe.to(distributed_state.device)
    with distributed_state.split_between_processes(data_shards) as dataset_shard:
        string_predictions.extend(sqlcoder_generation(dataset_shard, pipe=pipe))

    # calculate metric
    processed_predictions = post_processing_fn(string_predictions)
    processed_targets = post_processing_fn(created_datasets['test']['answers'])
    metric_outputs = metric_function(processed_predictions, processed_targets)
    # log metric result
    # only consider first returned value if metric has multiple return values, which is main result by convention (others are supplemental information)
    metric_results = {f"test/{metric_function.__name__}": metric_outputs[0] if isinstance(metric_outputs, tuple) else metric_outputs}

    print(metric_results)
    if wandb_logger is not None:
        wandb_logger.log_dict(
                metric_results,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(string_predictions),
                )

        text_predictions = [[pred] for pred in string_predictions]
        wandb_logger.log_table(key='text_predictions', columns=['text_predictions'], data=text_predictions)
    else:
        predictions_save_path = os.path.join(
            data_args.data_dir,
            f"sqlcoder_{data_args.table_corpus}_{data_args.dataset_name or '-'.join(data_args.template_names)}_test_predictions.txt"
            )
        with open(predictions_save_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}\n")
            f.write(f"Metric results: {metric_results}\n")
            prediction_lines = '\n'.join(string_predictions)
            f.write(f"Predictions:\n{prediction_lines}\n")
