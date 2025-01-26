import os
from datetime import datetime
from typing import List, Optional, Union

import datasets
import lightning as L
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.utils.sql_utils import execute_sql


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


def get_sqlcoder_tokenizer(hf_version_path: str = "defog/sqlcoder-7b-2", **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(hf_version_path, **kwargs)
    return tokenizer


def get_sqlcoder_model(hf_version_path: str = "defog/sqlcoder-7b-2"):
    return AutoModelForCausalLM.from_pretrained(
        hf_version_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )


def get_sqlcoder_inference_pipeline(model=None, tokenizer=None, max_query_length: int = 300, num_beams: int = 5, device: Optional[str] = None):
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
        device=device,
        device_map="auto" if device is None else None,
    )


def run_pipeline(pipe, prompts: Union[List[str], KeyDataset], batch_size: int = 8) -> List[str]:
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


def sqlcoder_generation(question: Union[str, List[str], datasets.Dataset], table: Optional[Union[dict, List[dict]]] = None, model=None, tokenizer=None, pipe=None, batch_size: Optional[int] = None) -> Union[str, List[str]]:
    is_single_query = False
    if pipe is None:
        pipe = get_sqlcoder_inference_pipeline(model, tokenizer)

    if isinstance(question, datasets.Dataset):
        generated_query = run_pipeline(pipe, KeyDataset(question, 'questions'), batch_size=batch_size)
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
