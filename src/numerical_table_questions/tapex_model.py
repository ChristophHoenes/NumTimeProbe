import logging
import transformers
from pathlib import PurePath
from tqdm import tqdm
from typing import List

from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet
from numerical_table_questions.data_utils import cast_to_reduced_int
from numerical_table_questions.metrics import str_match_accuracy


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def tapex_model_type_info() -> dict:
    return dict(
        model_name_or_path='tapex',
        pad_token_id=1,
        mask_token_id=-100,
        input_targets=True,
        loss_out_id='loss',
        filter_data_attributes=['input_ids', 'attention_mask'],
        tensor_input_mapping={
            '*': None,
            'labels': lambda x, y: y,
            },
        dict_input_mapping={
                    'input_ids': 'input_ids',
                    'attention_mask': 'attention_mask',
                    'labels': 'targets',
                    },
        )


def tapex_model():
    model = transformers.BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")
    # potentially change model config
    # model.config.xxx = 'xxx'
    return model


def tapex_config() -> dict:
    config = dict(
        generation_metrics={
            'str_match_accuracy': (str_match_accuracy,
                                   {
                                    'post_processing_fn': lambda x: [item.strip() for item in x],
                                    }
                                   )
            }
        )
    return config


def tapex_tokenizer_format(data, lazy: bool = False, **kwargs):
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


def tapex_tokenize(tokenizer, tokenizer_inputs, pad_token_id, mask_token_id, verbose, **kwargs) -> List[dict]:
    if verbose:
        progress_bar = tqdm(tokenizer_inputs)
        progress_bar.set_description("Tapex Tokenizing (inputs and targets separately)...")
    else:
        progress_bar = tokenizer_inputs  # no progress bar

    tokenized_tuples = [
        (table_question['query'], tokenizer(**table_question), target['answer'], tokenizer(**target)['input_ids'])
        for table_question, target in progress_bar
        ]
    # convert list of tuples to list of dicts (add target ids to input's tokenized dict)
    tokenized = []
    for question, input_tokenized, answer, target_input_ids in tokenized_tuples:
        # save original text along with tokenized input
        input_tokenized['questions'] = question
        input_tokenized['answers'] = answer
        # add target padding mask (e.g. swap pad tokens with padding mask token e.g. -100 for BART)
        target_input_ids[target_input_ids == pad_token_id] = mask_token_id
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
    return tokenized
