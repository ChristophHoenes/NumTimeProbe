from pathlib import Path
from typing import Union, Dict, Tuple

import datasets
import torch

from numerical_table_questions.data_caching import caching
from numerical_table_questions.data_utils import cutoff_num_questions
from numerical_table_questions.data_synthesis import Table
from numerical_table_questions.tokenizer_utils import get_tokenizer, prepare_for_tokenizer, model_specific_tokenizing, restore_metadata, convert_to_long_tensor_if_int_tensor


def generate_question_index(table_dataset) -> Dict[int, Tuple[str, int]]:
    """Given a TableQuestionDataset compute a mapping from question index to table id and question index within the table."""
    idx = -1
    question2table_index = {(idx := idx + 1): (t, q)
                            for t, table in enumerate(table_dataset)
                            for q, _ in enumerate(table['questions'])
                            }
    return question2table_index


class QuestionTableIndexDataset(torch.utils.data.Dataset):
    def __init__(self, table_dataset: Union[str, Path, datasets.Dataset], data_dir: str = './data/NumTabQA/.cache', cutoff=None):
        if isinstance(table_dataset, (str, Path)):
            self.data_dir = data_dir
            self.dataset_version = table_dataset
            table_dataset = caching(self.dataset_version, cache_path=data_dir)
            if table_dataset is None:
                raise FileNotFoundError(f"No table dataset was found at path {self.data_dir + '/' + self.dataset_version}")
        else:
            self.data_dir = None
            self.dataset_version = None
        if cutoff is not None:
            table_dataset = cutoff_num_questions(table_dataset, cutoff=cutoff)

        self.index_dict = generate_question_index(table_dataset)
        self.table_dataset = table_dataset

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        table_idx, question_number = self.index_dict[idx]
        table_data = self.table_dataset.select([table_idx])
        # maybe leave for collate for more flexibility
        # table = Table.from_state_dict(table_data['table'])
        # question = table_data['questions'][question_number]
        return {'table_data': table_data, 'table_idx': table_idx, 'question_number': question_number, 'question_id': idx}


def table_collate(batch_of_index_ids, model_name, tokenizer, tokenizing_args,
                  pad_token_id: int, mask_token_id: int, truncation='drop_rows_to_fit', padding='max_length',
                  is_eval: bool = False,
                  ):
    tokenized_batch = []
    # get table and question from dataset
    for sample_idx in batch_of_index_ids:
        table_data = sample_idx['table_data']
        question_number = sample_idx['question_number']
        # table augmentation
        # apply informed column deletion (only unaffected columns are dropped)
        # apply informed row deletion (delete percentage of rows that are unaffected by condition, infer from answer_coordinates)
        # apply column permutation
        # apply row permutation
        tokenizer_inputs = []
        tokenizer_inputs.extend(
            prepare_for_tokenizer(table_data, model_name=model_name, lazy=True, question_number=question_number, truncation=truncation, padding=padding, is_eval=is_eval)
            )
        tokenized_dict = model_specific_tokenizing(tokenizer, tokenizer_inputs, model_name, pad_token_id=pad_token_id, mask_token_id=mask_token_id, verbose=False, **tokenizing_args)
        # save tokenizer keys for default filter of keys in dict style batch
        tokenizer_keys = list(tokenized_dict.keys())
        # add all keys that were present in the original dataset (excluding the table to conserve memory)
        restore_metadata(table_data, tokenized_dict, question_number)
        # add tokenizer key names as field to be able to only select those in model forward pass
        tokenized_dict.update({'tokenizer_keys': [tokenizer_keys]})  # by convention each field is two-level iterable (list of lists or list of tensor with one batch dimension at 0)
        tokenized_batch.append(tokenized_dict)

    tokenizer_output_names = tokenized_batch[0].keys()  # all samples should have the same tokenizer outputs
    # check for each keys/fields the value is of tensor type (for all samples in the batch)
    is_all_tensors = {key: all([isinstance(sample[key][0], torch.Tensor) for sample in tokenized_batch]) for key in tokenizer_output_names}
    # concat tensors at batch dimension (from list of single-sample-tensor-lists to tensor batch; list of values for non-tensor fields)
    tokenized_batch = {key: (torch.cat([sample[key][0] for sample in tokenized_batch])  # concat tensors of samples if possible
                             if is_all_tensors[key]
                             else [sample[key][0][0]  # if sample's field is list with single item, unwrap it from the list
                                   if isinstance(sample[key][0], list) and len(sample[key][0]) == 1
                                   else sample[key][0]  # unwrap table batch dimension
                                   for sample in tokenized_batch
                                   ]  # if field is not tensor type (int, str, list of multiple items) return a list of samples' values
                             )
                       for key in tokenizer_output_names
                       }
    # test consistency of non-metadata keys within batch and reduce tokenizer_keys to list of strings (instead of list of lits of strings)
    if any([sample != tokenized_batch['tokenizer_keys'][0] for sample in tokenized_batch['tokenizer_keys']]):
        raise ValueError("All samples of a batch must have the same keys/fields!")
    else:
        tokenized_batch['tokenizer_keys'] = tokenized_batch['tokenizer_keys'][0]  # same keys in all samples -> only one list for entire batch

    # add additional fields as meta info
    # if no padding idx is present -> add ooc (out of context) label for test set performance insights
    # TODO is sep token present when truncated? is this a problem? when comparing for pad_token_id perfect fit examples get mislabeled
    tokenized_batch['is_truncated'] = tokenized_batch['input_ids'] != pad_token_id  # if not ending with pad token assume truncation
    # for simplifying debugging include table / question id
    tokenized_batch['question_id'] = torch.LongTensor([sample['question_id'] for sample in batch_of_index_ids])  # global question id
    tokenized_batch['table_id'] = [sample['table_data'][0]['table']['table_id'] for sample in batch_of_index_ids]
    tokenized_batch['table_idx'] = torch.LongTensor([sample['table_idx'] for sample in batch_of_index_ids])  # for index based access directly in QuestionTableIndexDataset
    tokenized_batch['question_number'] = torch.LongTensor([sample['question_number'] for sample in batch_of_index_ids])  # local id within table batch

    # ensure all int type tensors have dtype long
    tokenized_batch = {key: convert_to_long_tensor_if_int_tensor(value)
                       for key, value in tokenized_batch.items()
                       }
    return tokenized_batch


if __name__ == "__main__":
    # load dataset, get index and tokenizer
    table_question_dataset = datasets.Dataset.load_from_disk('./data/NumTabQA/.cache/count_wikitables_validation_filtered_multi_answer_filter_agg_count_0/240425_1315_37_817267')
    data_by_table_id = QuestionTableIndexDataset(table_question_dataset)
    tokenizer = get_tokenizer("tapas")

    dataloader = torch.utils.data.DataLoader(
        # list(index.keys()),
        data_by_table_id,  # QuestionTableIndexDataset instead of index iterable
        batch_size=64, shuffle=True,
        # pass tokenizer and other parameters to collate function
        collate_fn=lambda x: table_collate(x, 'tapas', tokenizer, truncation='drop_rows_to_fit', padding='max_length')
        )

    for batch in dataloader:
        print(batch['input_ids'][0])
