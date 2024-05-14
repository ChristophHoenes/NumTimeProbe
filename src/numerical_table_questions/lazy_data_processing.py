from pathlib import Path
from typing import Union, Dict, Tuple

import datasets
import torch

from numerical_table_questions.data_synthesis import Table
from numerical_table_questions.tokenizer_utils import get_tokenizer, prepare_for_tokenizer


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
        if isinstance(table_dataset, str, Path):
            self.path = table_dataset
            table_dataset = datasets.Dataset.load_from_disk(self.path )
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


def processing_steps(batch_of_index_ids, tokenizer, truncation='drop_rows_to_fit', padding='max_length'):
    tokenized = []
    # get table and question from dataset
    for sample_idx in batch_of_index_ids:
        table_data = sample_idx['table_data']
        question_number = sample_idx['question_number']
        # table augmentation
        # apply informed column deletion (only unaffected columns are dropped)
        # apply informed row deletion (delete percentage of rows that are unaffected by condition, infer from answer_coordinates)
        # apply column permutation
        # apply row permutation
        tokenizer_inputs = prepare_for_tokenizer([table_data], question_number=question_number, truncation=truncation, padding=padding)
        table = Table.from_state_dict(table_data['table'])
        table_df = table.pandas_dataframe
        # retrieve question
        question = table_data['questions'][question_number]
        answer_coordinates = [(0, 0)]  # TODO remove dummy after answer_coordinates is implemented during synthesis
        # tokenize (auto truncate, pad to max seq len)
        tokenized.append(tokenizer(table=table_df, queries=[question], answer_coordinates=[answer_coordinates], answer_text=table_data["answers"][question_number], truncation=truncation, padding=padding, return_tensors="pt"))
        # ooc (out of context label for test set)
        # if no padding idx -> add label ooc (is sep present when truncated?)
    # concat at batch dimension
    tokenizer_output_names = tokenized[0].keys()
    tokenized_batch = {key: torch.cat([sample[key] for sample in tokenized]) for key in tokenizer_output_names}
    pad_token_id = 0
    is_truncated_feature = [sample['input_ids'][:, -1] != pad_token_id for sample in tokenized]
    tokenized_batch['is_truncated'] = torch.BoolTensor(is_truncated_feature)
    # for simplifying debugging include question_id
    question_id = torch.IntTensor([sample['question_id'] for sample in batch_of_index_ids])
    tokenized_batch['question_id'] = question_id
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
        collate_fn=lambda x: processing_steps(x, tokenizer, truncation='drop_rows_to_fit', padding='max_length')
        )

    for batch in dataloader:
        print(batch['input_ids'][0])
