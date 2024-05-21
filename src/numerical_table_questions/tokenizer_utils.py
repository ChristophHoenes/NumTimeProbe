from collections.abc import Iterable
from typing import Union, Optional, List, Dict

import torch
from tqdm import tqdm
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
        if not lazy:  # do not spam logger with messages every batch (lazy processing during data loading)
            logger.info("Flattening examples to tokenizer format...")
        padding = kwargs.get('padding') or True
        truncation = kwargs.get('truncation') or True
        max_length = kwargs.get('max_length') or 1024
        return_tensors = kwargs.get('return_tensors') or 'pt'
        if isinstance(data, TableQuestionDataSet):
            if lazy:
                raise NotImplementedError("No processing implemented for lazy oprion and non-huggingface-datasets serialization!")
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
            if lazy:
                # extract only single sample at specific question_number from a table batch
                return [
                    (
                        {
                            'table': Table.from_state_dict(table_batch['table']).pandas_dataframe,
                            'query': table_batch['questions'][kwargs['question_number']],
                            'padding': padding,
                            'truncation': truncation,
                            'max_length': max_length,
                            'return_tensors': return_tensors,
                        },
                        {
                            'answer': table_batch['answers'][kwargs['question_number']],
                            'padding': padding,
                            'truncation': truncation,
                            'max_length': max_length,
                            'return_tensors': return_tensors,
                        }
                    )
                    for table_batch in data
                ]
            else:
                data_iterable = tqdm(data)
                data_iterable.set_description("transfer to TAPEX tokenizer format...")
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
                    for table_batch in data_iterable
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


def cast_to_reduced_int(ints: torch.Tensor, num_values: Optional[int] = None):
    """
        Selects the smallest possible torch dtype for ints representing an id mapping of size num_value.
        If num_values is None the amount of values (e.g. vocab size) is estimated by the maximum of the
        values in the tensor plus one (for id zero).
    """
    # if num_values is None infer the coding size
    if num_values is None:
        num_values = ints.max() + 1
    if num_values <= 2:
        cast_to = torch.bool
    elif num_values <= 128:
        cast_to = torch.int8
    elif num_values <= 256 and ints.min() >= 0:
        cast_to = torch.uint8
    elif num_values <= 32768:
        cast_to = torch.int16
    elif num_values <= 2_147_483_648:
        cast_to = torch.int32
    else:
        cast_to = torch.int64
    return ints.to(cast_to)


def list_of_dict_2_dict_of_lists(list_of_dicts: List[dict]) -> Dict[str, list]:
    if len(list_of_dicts) == 0:
        return dict()
    keys = list_of_dicts[0].keys()
    if any([sample.keys() != keys for sample in list_of_dicts]):
        raise KeyError("Encountered mismatch between keys of different dictionaries!")
    return {key: [sample[key] for sample in list_of_dicts] for key in keys}


def model_specific_tokenizing(tokenizer, tokenizer_inputs: Union[List[dict], dict], model_name: str, **kwargs):
    if model_name == 'tapex':
        progress_bar = tqdm(tokenizer_inputs)
        progress_bar.set_description("Tapex Tokenizing (inputs and targets separately)...")
        tokenized_tuples = [
            (tokenizer(**table_question), tokenizer(**target)['input_ids'])
            for table_question, target in progress_bar
            ]
        # convert list of tuples to list of dicts (add target ids to input's tokenized dict)
        tokenized = []
        for input_tokenized, target_input_ids in tokenized_tuples:
            input_tokenized['answers'] = target_input_ids
            tokenized.append(input_tokenized)

        """
        apply_sequence_transform(tokenized
                             transform_fn=unbind_table_batch,
                             field_names=['input_ids'],  # apply to all keys (assuming all tokenizer outputs have the same keys)
                             pad_token_id=,
                       end_of_sequence_idxs=None,
                       sequence_dimension=1,
                             )
        apply_sequence_transform(tokenized
                             transform_fn=unbind_table_batch,
                             field_names=list(tokenized[0].keys()),  # apply to all keys (assuming all tokenizer outputs have the same keys)
                             pad_token_id=,
                       end_of_sequence_idxs: Optional[List[int]] = None,
                       sequence_dimension: int = 1,
                             )
        # flatten the table batches to sequence of questions
        tokenized_dict = unbind_table_batch(tokenized, self.model_specs.pad_token_id)
        """
        if kwargs.get('optimize_int_type'):
            # infer smallest possible int dtype to represent the ids
            tokenized = [
                sample | {  # keep all keys of sample but update the following
                    'input_ids': cast_to_reduced_int(sample['input_ids'], num_values=tokenizer.vocab_size),
                    'attention_mask': cast_to_reduced_int(sample['attention_mask'], num_values=2),  # reduce to binary int format for mask
                    'answers': cast_to_reduced_int(sample['answers'], num_values=tokenizer.vocab_size),
                    }
                for sample in tokenized
                ]
        # convert to dict of lists
        return list_of_dict_2_dict_of_lists(tokenized)
    else:
        logger.info(f"No custom tokenizing procedure for model '{model_name}'. Using standard tokenizer call.")
        progress_bar = tqdm(tokenizer_inputs)
        progress_bar.set_description("Generic Tokenization...")
        if isinstance(tokenizer_inputs, list):  # sample by sample
            tokenized = [tokenizer(**sample) for sample in tokenizer_inputs]
        else:  # batched
            tokenized = tokenizer(**tokenizer_inputs)
        if kwargs.get('optimize_int_type'):
            # infer smallest possible int dtype to represent the ids
            tokenized = [
                sample | {  # keep all keys of sample but update the following
                    'input_ids': cast_to_reduced_int(sample['input_ids'], num_values=tokenizer.vocab_size)
                    }
                for sample in tokenized
                ]
        return list_of_dict_2_dict_of_lists(tokenized)
