import warnings
from collections.abc import Iterable, Callable
from time import time
from typing import Union, Optional, List, Dict, Tuple

import datasets
import torch
from tqdm import tqdm
from loguru import logger  # TODO check difference to custom python logger and maybe change
from transformers import TapexTokenizer, TapasTokenizer
from transformers.models.auto.tokenization_auto import AutoTokenizer

from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet
from numerical_table_questions.utils.data_utils import cast_to_reduced_int
from numerical_table_questions.utils.model_utils import extract_model_name
from numerical_table_questions.sqlcoder_model import get_sqlcoder_tokenizer
from numerical_table_questions.tapex_model import tapex_tokenizer_format, tapex_tokenize
from numerical_table_questions.tapas_model import tapas_tokenizer_format, reduce_answer_coordinates


def get_tokenizer(model_name_or_path: str, **kwargs):
    model_name = extract_model_name(model_name_or_path)
    match model_name.lower():
        case 'tapex':
            return TapexTokenizer.from_pretrained(model_name_or_path if '/' in model_name_or_path else "microsoft/tapex-base-finetuned-wtq", **kwargs)
        case 'omnitab':
            return AutoTokenizer.from_pretrained(model_name_or_path if '/' in model_name_or_path else "neulab/omnitab-large-finetuned-wtq", **kwargs)
        case 'tapas':
            return TapasTokenizer.from_pretrained(model_name_or_path if '/' in model_name_or_path else "google/tapas-base", **kwargs)
        case 'sqlcoder':
            return get_sqlcoder_tokenizer(model_name_or_path if '/' in model_name_or_path else "defog/sqlcoder-7b-2", **kwargs)
        case 'reastap':
            return AutoTokenizer.from_pretrained(model_name_or_path if '/' in model_name_or_path else "Yale-LILY/reastap-large", **kwargs)
        case _:
            # TODO proper logging
            print(f"No tokenizer explicitly implemented for model '{model_name_or_path}'. Trying to load tokenizer from Huggingface model hub...")
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)


def prepare_for_tokenizer(data: Union[TableQuestionDataSet, Iterable[dict]], model_name_or_path: str, lazy: bool = False, is_eval: bool = False, **kwargs):
    model_name = extract_model_name(model_name_or_path)
    match model_name.lower():
        case 'tapex' | 'omnitab' | 'reastap':
            return tapex_tokenizer_format(data, lazy, **kwargs)
        case 'tapas':
            return tapas_tokenizer_format(data, lazy, is_eval, **kwargs)
        case _:
            raise NotImplementedError(f"No tokenization behavior implemented for model '{model_name}'!")


def convert_to_long_tensor_if_int_tensor(feature):
    """ Auxilary function to ensure long type Tensors as model inputs.
        Useful when loading potentially reduced int types (see cast_to_reduced_int) from disk.
        Function allows for any input type and only modifies the input if it is an int tensor.
        Otherwise the input is returned unchanged. If the tensor is already of type long conversion is skipped as well.
    """
    if isinstance(feature, torch.Tensor) and not torch.is_floating_point(feature) and not isinstance(feature, torch.LongTensor):
        original_device = feature.device
        converted_tensor = feature.type(torch.LongTensor).to(original_device)
        return converted_tensor
    else:
        return feature  # if not a tensor or float return original value


def list_of_dicts_2_dict_of_lists(list_of_dicts: List[dict]) -> Dict[str, list]:
    if len(list_of_dicts) == 0:
        return dict()
    keys = list_of_dicts[0].keys()
    if any([sample.keys() != keys for sample in list_of_dicts]):
        raise KeyError("Encountered mismatch between keys of different dictionaries!")
    return {key: [sample[key] for sample in list_of_dicts] for key in keys}


def get_attention_mask(batch, pad_token_id: int = 1, padding_mask_token_id: int = 0, normal_token_id: int = 1):
    attention_mask_template = torch.ones_like(batch) * normal_token_id
    is_padding_mask = batch == pad_token_id
    attention_mask_template[is_padding_mask] = padding_mask_token_id
    return attention_mask_template


def default_tokenize(tokenizer, tokenizer_inputs, model_name, verbose, **kwargs):
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
    return tokenized


def model_specific_tokenizing(tokenizer, tokenizer_inputs: Union[List[dict], dict],
                              model_name_or_path: str,
                              # TODO maybe infer from model type info but maybe better instance-wise -> more direct/safe
                              pad_token_id: int,
                              mask_token_id: int,
                              verbose: bool = True,
                              **kwargs):
    # if a single sample (dict) is passed temporarily wrap in list for generalization
    if isinstance(tokenizer_inputs, dict):
        tokenizer_inputs = [tokenizer_inputs]
        single_example = True
    else:
        single_example = False
    model_name = extract_model_name(model_name_or_path)
    match model_name.lower():
        case 'tapex' | 'omnitab':
            tokenized = tapex_tokenize(tokenizer, tokenizer_inputs, pad_token_id, mask_token_id, verbose, **kwargs)
        case 'tapas':
            # reduce answer coorndinates for tables that do not fit into the model
            reduce_answer_coordinates(tokenizer, tokenizer_inputs)
            """
            for tok_input in tokenizer_inputs:
                if tok_input.get('answer_coordinates') is not None:
                    iteration = 0
                    while len(tok_input['answer_coordinates'][0]) > 0:
                        try:
                            tokenizer(**tok_input)
                            break  # if tokenization successful leave loop
                        except ValueError as e:
                            logger.debug(e)
                            logger.debug(f"Table does not fit reducing answer coordinates iteration {iteration}...")
                            reduce_answer_coordinates(tok_input, iteration, max_length=512)
                            iteration += 1
            """
            tokenized = default_tokenize(tokenizer, tokenizer_inputs, pad_token_id, verbose, **kwargs)
            # add float answer (required for weak supervision)
            for encoding, tok_input in zip(tokenized, tokenizer_inputs):
                try:
                    float_answer = float(tok_input['answer_text'][0])
                    encoding['float_answer'] = torch.tensor(float_answer).unsqueeze(dim=0)
                except (TypeError, ValueError):
                    if tok_input['answer_text'] is None:
                        warnings.warn("Answer text is None! The float answer will be None.")
                    else:
                        warnings.warn(f"Could not convert answer text '{tok_input['answer_text']}' to float! The float answer will be None.")
                    encoding['float_answer'] = torch.tensor(float('nan')).unsqueeze(dim=0)
                if token_type_size := kwargs.get('token_type_size'):
                    token_type_tensor = encoding['token_type_ids']
                    # replace token_type_ids that are out of range of the embedding matrix with the highest possible index
                    token_type_tensor[token_type_tensor >= token_type_size] = token_type_size - 1
            # add attention_mask
            #print('default_tokenizer_keys', list(tokenized[0].keys()))
            #for sample in tokenized:
            #    sample.update({'attention_mask': get_attention_mask(sample['input_ids'], pad_token_id=pad_token_id, padding_mask_token_id=0, normal_token_id=1)})
            #print('tapas_keys', list(tokenized[0].keys()))
        case _:
            tokenized = default_tokenize(tokenizer, tokenizer_inputs, pad_token_id, verbose, **kwargs)
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


def restore_metadata(original_data: Union[datasets.Dataset, List[dict]], tokenized_data: dict, question_number: Optional[int] = None):
    # check assumptions on inputs
    if not isinstance(tokenized_data['input_ids'], list):
        raise TypeError("Expected tokenized_data's values to be (nested) lists (table_batch and samples per table) "
                        f"but found {type(tokenized_data['input_ids'])} for tokenized_data['input_ids']!"
                        )
    if len(original_data) != len(tokenized_data['input_ids']):  # assumes that all possible tokenizers used have at least input_ids as output key
        raise ValueError(f"The number of samples in the original_data ({len(original_data)}) and tokenized_data ({len(tokenized_data['input_ids'])}) "
                         "must match in order to restore the metadata from original_data correctly!"
                         )

    # add other fields of data split that did not go through the tokenizer
    missing_fields = list(set(original_data[0].keys())
                          # TODO remove obsolete fields ('column_name', 'aggregator', 'condition_value') from data and this code
                          - set(['table', 'column_name', 'aggregator', 'condition_value'])  # do not copy table for each question
                          - set(tokenized_data.keys())
                          )
    additional_fields_dict = {field: [[table_batch[field][question_number]]  # if idx is specified only select value for a single sample
                                      if question_number is not None and isinstance(table_batch[field], list)
                                      else table_batch[field]  # otherwise select the entire table batch
                                      for table_batch in original_data
                                      ]
                              for field in missing_fields
                              }
    # since table was removed from keys for efficiency, add column with table_id for reference (repeat id for each question to the table)
    additional_fields_dict['table_id'] = [[table_batch['table']['table_id']] * len(table_batch['questions'])
                                          for table_batch in original_data
                                          ]
    # TODO table num rows, table num cols, unique values aggregation column
    tokenized_data.update(additional_fields_dict)
