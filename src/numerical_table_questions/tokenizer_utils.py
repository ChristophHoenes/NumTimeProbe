import warnings
from collections.abc import Iterable, Callable
from typing import Union, Optional, List, Dict, Tuple

import datasets
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


def list_of_dicts_2_dict_of_lists(list_of_dicts: List[dict]) -> Dict[str, list]:
    if len(list_of_dicts) == 0:
        return dict()
    keys = list_of_dicts[0].keys()
    if any([sample.keys() != keys for sample in list_of_dicts]):
        raise KeyError("Encountered mismatch between keys of different dictionaries!")
    return {key: [sample[key] for sample in list_of_dicts] for key in keys}


def model_specific_tokenizing(tokenizer, tokenizer_inputs: Union[List[dict], dict],
                              model_name: str,
                              # TODO maybe infer from model type info but maybe better instance-wise -> more direct/safe
                              pad_token_id: int, mask_token_id: int,
                              verbose: bool = True, **kwargs):
    # if a single sample (dict) is passed temporarily wrap in list for generalization
    if isinstance(tokenizer_inputs, dict):
        tokenizer_inputs = [tokenizer_inputs]
        single_example = True
    else:
        single_example = False

    if model_name == 'tapex':
        if verbose:
            progress_bar = tqdm(tokenizer_inputs)
            progress_bar.set_description("Tapex Tokenizing (inputs and targets separately)...")
        else:
            progress_bar = tokenizer_inputs  # no progress bar

        tokenized_tuples = [
            (tokenizer(**table_question), tokenizer(**target)['input_ids'])
            for table_question, target in progress_bar
            ]
        # convert list of tuples to list of dicts (add target ids to input's tokenized dict)
        tokenized = []
        for input_tokenized, target_input_ids in tokenized_tuples:
            # add field for tokenized targets
            input_tokenized['targets'] = target_input_ids
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
                    'targets': cast_to_reduced_int(sample['targets'], num_values=tokenizer.vocab_size),
                    }
                for sample in tokenized
                ]
    else:
        if verbose:
            logger.info(f"No custom tokenizing procedure for model '{model_name}'. Using standard tokenizer call.")
            progress_bar = tqdm(tokenizer_inputs)
            progress_bar.set_description("Generic Tokenization...")
        else:
            progress_bar = tokenizer_inputs  # no progress bar

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
    # convert from list of samples (dicts) to dict (samples as lists grouped by key)
    tokenized = list_of_dicts_2_dict_of_lists(tokenized)
    if single_example:
        tokenized = {key: value[0]  # unwrap list; extract first (and only) sample
                     for key, value in tokenized.items()
                     }
    return tokenized


def batch_end_of_sequence(batch: torch.Tensor, pad_token_id: int, sequence_dimension: int = 1) -> torch.Tensor:
    """ Returns the indices where the pad token occurs for the first time. """
    is_padding = batch == pad_token_id
    any_padding = is_padding.sum(dim=sequence_dimension) >= 1
    first_padding = is_padding.int().argmax(dim=sequence_dimension)
    return torch.where(any_padding, first_padding, batch.shape[-1])


def apply_sequence_transform(sequence_data:  Union[torch.Tensor, List[torch.Tensor], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]],
                             transform_fn: Callable[[torch.Tensor], torch.Tensor],
                             field_names: Optional[List[str]] = None,
                             verbose: bool = True,
                             **kwargs
                             ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
    """ Applies a sequence transform to every tensor in seqence_data.
        Has side effects on seqence_data if it is a dictionary (changes contents at key=field_names in place).
    """
    if isinstance(sequence_data, dict):
        if field_names is None:
            raise ValueError("Must specify to which fields (dict keys) the transform should be applied! But field_names was None, expected list of strings.")
        for field in field_names:
            if verbose:  # possibility to disable logging for many consecutive calls
                logger.info(f"Processing field '{field}':")
            sequence_data[field] = transform_fn(sequence_data[field], verbose=verbose, **kwargs)
        return sequence_data
    else:
        return transform_fn(sequence_data, **kwargs)


def unbind_table_batch(bound_table_batches: Union[torch.Tensor, List[torch.Tensor]],
                       pad_token_id: int,
                       end_of_sequence_idxs: Optional[List[int]] = None,
                       keep_batch_dim: bool = True,
                       sequence_dimension: int = 1,
                       verbose: bool = True,
                       ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """ Unbind single questions and aswers from a table batch and remove batch padding. """
    if verbose:  # possibility to disable logging for many consecutive calls
        logger.info("Unbinding table batches...")
    # wrap in list if single tensor (only one table batch)
    if isinstance(bound_table_batches, torch.Tensor):
        bound_table_batches = [bound_table_batches]
    # if the end of sequence ids are not provided calculate them based on the first occurance of padding token
    if end_of_sequence_idxs is None:
        end_of_sequence_idxs = [batch_end_of_sequence(batch, pad_token_id, sequence_dimension) for batch in bound_table_batches]
    # flatten rows (questions) in each (table) batch and discard everything after the respective entry in end_of_sequence_idxs (first pad token)
    batch_unbound = [row[:end_of_sequence_idxs[b][r]].unsqueeze(dim=0)
                     if keep_batch_dim
                     else row[:end_of_sequence_idxs[b][r]]
                     for b, batch in enumerate(bound_table_batches)
                     for r, row in enumerate(batch)
                     ]
    return batch_unbound, end_of_sequence_idxs


def truncate(tokenized: Union[torch.Tensor, List[torch.Tensor]],
             truncation_type: Union[bool, str],
             max_sequence_length: int,
             query_first: bool = True,
             num_reserved: Optional[Union[int, Iterable[Iterable[int]]]] = None,
             sequence_dimension: int = 1,
             verbose: bool = True,
             ) -> Union[torch.Tensor, List[torch.Tensor]]:
    if verbose:  # possibility to disable logging for many consecutive calls
        logger.info(f"Applying truncation strategy '{truncation_type}'...")
    # wrap in list if single tensor (only one table batch)
    # TODO maybe use decorator for (un)wrapping
    single_tensor = False
    if isinstance(tokenized, torch.Tensor):
        single_tensor = True
        tokenized = [tokenized]
    # determine how many steps in each sequence to reserve for additional input
    if num_reserved is None:
        num_reserved = 0
    if isinstance(num_reserved, int):
        num_reserved = [[num_reserved] * batch.shape[-1] for batch in tokenized]
    # TODO option to drop tokens using heuristic such that question is still solvable (e.g drop unused column)
    if truncation_type in (True, 'longest_first'):
        truncated = [table_question[:(max_sequence_length - num_reserved[tq])]  # truncate from right
                     if query_first
                     else table_question[-(max_sequence_length - num_reserved[tq]):]  # truncate from left
                     for tq, table_question in enumerate(tokenized)
                     ]
    elif truncation_type in (False, 'do_not_truncate'):
        # filter out sequences larger than model's max_length
        truncated = [
            table_question
            for b in range(len(tokenized))
            for i, table_question in enumerate(tokenized)
            if tokenized[b][i].shape[sequence_dimension] + num_reserved[b][i] <= max_sequence_length
        ]
    else:
        # for compatibility with huggingface implement only_first & only_second but not needed up to now
        # TODO think about value error and check allowed argument values?
        raise NotImplementedError(f"Truncation strategy '{truncation_type}' is not implemented!")
    if single_tensor:
        return truncated[0]
    return truncated


def pad(tokenized: Union[torch.Tensor, List[torch.Tensor]],
        padding_type: Union[bool, str],
        max_sequence_length: int,
        pad_token_id: int,
        sequence_dimension: int = 1,
        verbose: bool = True,
        ):
    if verbose:  # possibility to disable logging for many consecutive calls
        logger.info(f"Applying padding strategy '{padding_type}'...")
    # wrap in list if single tensor (only one table batch)
    # # TODO maybe use decorator for (un)wrapping
    single_tensor = False
    if isinstance(tokenized, torch.Tensor):
        single_tensor = True
        tokenized = [tokenized]
    if padding_type in (True, 'max_length', 'longest'):
        if padding_type in (True, 'max_length'):
            # infer shape of tokenized tensors (or table batches) and change the sequence_dimension
            # to the maximum context length of the model
            tensor_shape = list(tokenized[0].shape)
            tensor_shape[sequence_dimension] = max_sequence_length
            # append dummy tensor to ensure maximum length sequence is present
            tokenized.append(torch.zeros(tensor_shape))
        # pad to longest sequence and unbind to list of single samples again
        padded = list(
            torch.unbind(
                torch.nn.utils.rnn.pad_sequence(
                    [table_question.squeeze() for table_question in tokenized],
                    batch_first=True,
                    padding_value=float(pad_token_id),
                    )
                )
            )
        # for True and 'max_length' pop dummy tensor that was previously appended
        if padding_type in (True, 'max_length'):
            padded = padded[:-1]
    elif padding_type in (False, 'do_not_pad'):
        padded = tokenized
    else:
        raise NotImplementedError(f"Padding strategy '{padding_type}' is not implemented!")
    if single_tensor:
        return padded[0]
    return padded


def post_tokenizing(tokenized: dict, tokenizing_args: dict, max_sequence_length: int, pad_token_id: int,
                    mask_token_id: int, model_name: str, sequence_dimension: int = 1, verbose: bool = True):
    if model_name == 'tapex':
        # TODO maybe break down in reusable blocks for other models
        # first unbind (flatten) table batches to a list with one tensor per question (removes table batch padding)
        # save sizes of table batches for repeating constant features in tokenized
        table_batch_sizes = [table_batch.shape[0] for table_batch in tokenized['input_ids']]

        if verbose:
            logger.info("Processing field 'input_ids':")
        tokenized['input_ids'], end_seq_idxs = unbind_table_batch(tokenized['input_ids'],
                                                                  pad_token_id=pad_token_id,
                                                                  sequence_dimension=sequence_dimension,
                                                                  verbose=verbose,
                                                                  )
        # reuse computed end_of_sequence_idxs (row-wise last non-padding token) from 'input_ids' and apply to 'attention_mask'
        # they are inferred from pad_token_id which does not exist in masks, also saves computation
        if verbose:
            logger.info("Processing field 'attention_mask':")
        tokenized['attention_mask'], _ = unbind_table_batch(tokenized['attention_mask'],
                                                            pad_token_id=pad_token_id,
                                                            end_of_sequence_idxs=end_seq_idxs,
                                                            sequence_dimension=sequence_dimension,
                                                            verbose=verbose,
                                                            )
        # targets are tokenized separately from 'input_ids' and have their own end_of_sequence_idxs
        if verbose:
            logger.info("Processing field 'answers':")
        tokenized['answers'], _ = unbind_table_batch(tokenized['answers'],
                                                     # TODO rethink name mask token id <mask> token vs. mask target padding
                                                     pad_token_id=mask_token_id,  # targets are padded with padding mask token (e.g. -100 for BART)
                                                     sequence_dimension=sequence_dimension,
                                                     verbose=verbose,
                                                     )
        # flatten also additional keys in tokenized to allign with the tokenized fields
        for key in tokenized.keys():
            if key in ['input_ids', 'attention_mask', 'answers']:
                continue
            flattened = []
            for table_value, num_questions in zip(tokenized[key], table_batch_sizes):
                if isinstance(table_value, Iterable):
                    flattened.extend(table_value)
                else:
                    flattened.extend([table_value] * num_questions)
            tokenized[key] = flattened

        # save target lengths to determine token budget of input ids
        target_lengths = [tensor.shape[sequence_dimension] for tensor in tokenized['answers']]

        # if target does not fit into model warn and truncate target
        if any([target_len > max_sequence_length for target_len in target_lengths]):
            warnings.warn("Encountered target, that is greater than max_sequence_length! You might want to check your configuration.")
            if verbose:
                logger.info("Processing field 'answers':")
            truncated_targets = truncate(
                tokenized['answers'],
                truncation_type=tokenizing_args['truncation'],
                max_sequence_length=max_sequence_length,
                query_first=True,  # always truncate in the back for targets
                sequence_dimension=sequence_dimension,
                verbose=verbose,
                )
        else:
            truncated_targets = tokenized['answers']

        # apply same truncation to 'input_ids' and 'attention_mask' considering the target_lengths as well
        truncated_dict = apply_sequence_transform(
            sequence_data=tokenized,
            transform_fn=truncate,
            field_names=['input_ids', 'attention_mask'],
            truncation_type=tokenizing_args['truncation'],
            max_sequence_length=max_sequence_length,
            query_first=tokenizing_args['query_first'],
            # targets and input need to fit into the model at once
            num_reserved=target_lengths,
            sequence_dimension=sequence_dimension,
            verbose=verbose,
            )

        # fill up samples that are too short with padding
        if verbose:
            logger.info("Processing field 'input_ids':")
        padded_input_ids = pad(
            truncated_dict['input_ids'],
            padding_type=tokenizing_args['padding'],
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            sequence_dimension=sequence_dimension,
            verbose=verbose,
            )

        if verbose:
            logger.info("Processing field 'attention_mask':")
        padded_attention_mask = pad(
            truncated_dict['attention_mask'],
            padding_type=tokenizing_args['padding'],
            max_sequence_length=max_sequence_length,
            pad_token_id=0,  # fill up attention mask with zeros (where input has padding tokens)
            sequence_dimension=sequence_dimension,
            verbose=verbose,
            )

        if verbose:
            logger.info("Processing field 'answers':")
        padded_targets = pad(
            truncated_targets,
            padding_type=tokenizing_args['padding'],
            max_sequence_length=max_sequence_length,
            pad_token_id=mask_token_id,  # Bart for CLM required target to be 'padded' with mask_token_id (e.g. -100)
            sequence_dimension=sequence_dimension,
            verbose=verbose,
            )

        processed = {'input_ids': padded_input_ids if isinstance(padded_input_ids, torch.Tensor) else torch.cat(padded_input_ids, dim=0),
                     'attention_mask': padded_attention_mask if isinstance(padded_attention_mask, torch.Tensor) else torch.cat(padded_input_ids, dim=0),
                     'answers': padded_targets if isinstance(padded_targets, torch.Tensor) else torch.cat(padded_input_ids, dim=0),
                     }
        tokenized.update(processed)
    else:
        raise NotImplementedError(f"Custom post tokenization is not implemented for model '{model_name}'!")
    return tokenized


def restore_metadata(original_data: datasets.Dataset, tokenized_data: dict):
    # add other fields of data split that did not go through the tokenizer
    missing_fields = list(set(original_data.column_names)
                          # TODO remove obsolete fields ('column_name', 'aggregator', 'condition_value') from data and this code
                          - set(['table', 'column_name', 'aggregator', 'condition_value'])  # do not copy table for each question
                          - set(tokenized_data.keys())
                          )
    additional_fields_dict = {field: original_data[field] for field in missing_fields}
    # since table was removed from keys for efficiency, add column with table_id for reference (repeat id for each question to the table)
    additional_fields_dict['table_id'] = [[table_batch['table']['table_id']] * len(table_batch['questions'])
                                          for table_batch in original_data
                                          ]
    # TODO table num rows, table num cols, unique values aggregation column
    tokenized_data.update(additional_fields_dict)
