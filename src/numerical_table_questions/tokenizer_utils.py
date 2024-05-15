from collections.abc import Iterable
from typing import Union

import tqdm
from loguru import logger  # TODO check difference to custom python logger and maybe change
from transformers import TapexTokenizer, TapasTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer

from numerical_table_questions.data_synthesis import Table, TableQuestionDataSet


def get_tokenizer(model_name, **kwargs):
    if model_name == 'tapex':
        return TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
    elif model_name == 'omnitab':
        return AutoTokenizer.from_pretrained("neulab/omnitab-large-finetuned-wtq")
    elif model_name == 'tapas':
        return TapasTokenizer.from_pretrained("google/tapas-base")
    else:
        raise NotImplementedError(f"No tokenizer getter implemented for model '{model_name}'!")


def prepare_for_tokenizer(data: Union[TableQuestionDataSet, Iterable[dict]], model_name, lazy: bool = False, **kwargs):
    # TODO implement other models' specific pre-tokenizer formatting and default case
    # TODO maybe a dict of functions instead of long if/elif/else would be cleaner
    if model_name == 'tapex':
        # TODO try performance with list of tables (duplicated) references as well
        # TODO flatten each question to sample also considering alternative answers
        logger.info("Flattening examples to tokenizer format...")
        padding = kwargs.get('padding') or True
        truncation = kwargs.get('truncation') or True
        max_length = kwargs.get('max_length') or 1024
        return_tensors = kwargs.get('return_tensors') or 'pt'
        if isinstance(data, TableQuestionDataSet):
            if lazy:
                raise NotImplementedError("No processing implemented for lazy oprion and non-datasets serialization!")
            questions_by_table = {}
            for question in data._questions:
                if questions_by_table.get(question._table._table_id) is None:
                    questions_by_table[question._table._table_id] = {'questions': [question._nl_question],
                                                                     # TODO handle string conversion elsewhere
                                                                     'answers': [str(question._answer)]}
                else:
                    questions_by_table[question._table._table_id]['questions'].append(question._nl_question)
                    # TODO handle string conversion elsewhere
                    questions_by_table[question._table._table_id]['answers'].append(str(question._answer))
            return [
                (
                    {
                        'table': data._tables[table_id].pandas_dataframe,
                        'query': content_dict['questions'],
                        'padding': padding,
                        'truncation': truncation,
                        'max_length': max_length,
                        'return_tensors': return_tensors,
                    },
                    {
                        'answer': content_dict['answers'],
                        'padding': padding,
                        'truncation': truncation,
                        'max_length': max_length,
                        'return_tensors': return_tensors,
                    }
                )
                for table_id, content_dict in tqdm(questions_by_table.items())
            ]
        else:
            return [
                (
                    {
                        'table': Table.from_state_dict(table_batch['table']).pandas_dataframe,
                        'query': table_batch['questions'],
                        'padding': padding,
                        'truncation': truncation,
                        'max_length': max_length,
                        'return_tensors': return_tensors,
                    },
                    {
                        'answer': table_batch['answers'],
                        'padding': padding,
                        'truncation': truncation,
                        'max_length': max_length,
                        'return_tensors': return_tensors,
                    }
                )
                for table_batch in tqdm(data)
            ]
    elif model_name == 'tapas':
        if isinstance(data, TableQuestionDataSet):
            raise NotImplementedError("Preparation of TAPAS tokenizer format is only implemented for huggingface datasets serialization!")
        else:
            if lazy:
                processed_samples = []
                for sample in (progress_bar := tqdm(data)):
                    progress_bar.set_description("transfer to TAPAS tokenizer format...")
                    table = Table.from_state_dict(sample['table'])
                    table_df = table.pandas_dataframe
                    # retrieve question
                    question = sample['questions'][kwargs['question_number']]
                    answer_text = sample['answers'][kwargs['question_number']]
                    answer_coordinates = [(0, 0)]  # TODO remove dummy after answer_coordinates is implemented during synthesis
                    processed_samples.append({
                        'table': table_df,
                        'queries': [question],
                        'answer_coordinates': answer_coordinates,
                        'answer_text': [answer_text],
                        'padding': kwargs.get('padding', 'max_length'),
                        'truncation': kwargs.get('truncation', 'drop_rows_to_fit'),
                        'return_tensors': kwargs.get('return_tensors', 'pt'),
                    })
                return processed_samples
            else:
                raise NotImplementedError("No Tapas tokenizer preparation implemented for non-lazy option!")
    else:
        raise NotImplementedError(f"No tokenization behavior implemented for model '{model_name}'!")
