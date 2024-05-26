from pathlib import Path
from typing import Union, Dict, Tuple

import datasets
import torch

from numerical_table_questions.data_caching import caching
from numerical_table_questions.data_synthesis import Table
from numerical_table_questions.tokenizer_utils import get_tokenizer, prepare_for_tokenizer, model_specific_tokenizing, restore_metadata


def generate_question_index(table_dataset) -> Dict[int, Tuple[str, int]]:
    """Given a TableQuestionDataset compute a mapping from question index to table id and question index within the table."""
    idx = -1
    question2table_index = {(idx := idx + 1): (table['table']['table_id'], q)
                            for table in table_dataset
                            for q, _ in enumerate(table['questions'])
                            }
    return question2table_index


class QuestionTableIndexDataset(torch.utils.data.Dataset):
    def __init__(self, table_dataset: Union[str, Path, datasets.Dataset]):
        if isinstance(table_dataset, (str, Path)):
            self.path = table_dataset
            table_dataset = caching(self.path)
        else:
            self.path = None
        self.index_dict = generate_question_index(table_dataset)
        self.table_dataset = {sample['table']['table_id']: sample for sample in table_dataset}

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, idx):
        table_id, question_number = self.index_dict[idx]
        table_data = self.table_dataset[table_id]
        # maybe leave for collate for more flexibility
        # table = Table.from_state_dict(table_data['table'])
        # question = table_data['questions'][question_number]
        return {'table_data': table_data, 'question_number': question_number, 'question_id': idx}


def table_collate(batch_of_index_ids, model_name, tokenizer, tokenizing_args, pad_token_id: int, truncation='drop_rows_to_fit', padding='max_length'):
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
            prepare_for_tokenizer([table_data], model_name=model_name, lazy=True, question_number=question_number, truncation=truncation, padding=padding)
            )
        tokenized_dict = model_specific_tokenizing(tokenizer, tokenizer_inputs, model_name, **tokenizing_args)
        restore_metadata(table_data, tokenized_dict)
        tokenized_batch.append(tokenized_dict)
    # concat at batch dimension
    tokenizer_output_names = tokenized_dict[0].keys()
    is_all_tensors = {key: all([isinstance(sample[key], torch.Tensor) for sample in tokenized_batch]) for key in tokenizer_output_names}
    tokenized_batch = {key: (torch.cat([sample[key] for sample in tokenized_batch])  # concat tensors of samples if possible
                             if is_all_tensors[key]
                             else [sample[key]for sample in tokenized_batch]  # if field is not tensor type return a list of samples
                             )
                       for key in tokenizer_output_names
                       }

    # add additional fields as meta info
    # if no padding idx is present -> add ooc (out of context) label for test set performance insights
    # TODO is sep token present when truncated? is this a problem? when comparing for pad_token_id perfect fit examples get mislabeled
    tokenized_batch['is_truncated'] = tokenized_batch['input_ids'] != pad_token_id  # if not ending with pad token assume truncation
    # for simplifying debugging include table / question id
    tokenized_batch['question_id'] = torch.LongTensor([sample['question_id'] for sample in batch_of_index_ids])  # global question id
    tokenized_batch['table_id'] = [sample['table_data'][0]['table']['table_id'] for sample in batch_of_index_ids]
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
