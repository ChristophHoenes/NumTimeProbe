import os
import pickle

import datasets

from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset


PROMPT_TEMPLATE_QUESTION_FIRST = (
        "Here is a table with {col_sep_repr} as the delimiter between columns and a newline as the delimiter between rows. Answer the question based on the table.\n\n"
        'Question:\n"{question_text}"\n'
        "Table:\n"
        "{table_text}\n"
    )

PROMPT_TEMPLATE_TABLE_FIRST = (
        "Table:\n"
        "{table_text}\n"
        "Note: {col_sep_repr} is the delimiter between columns; {row_sep_repr} is the delimiter between rows.\n"
        'Question: "{question_text}"\n'
        "Answer the question based on the table.\n"
    )

def process_docs(dataset: datasets.Dataset):
    dataset = QuestionTableIndexDataset(dataset, lm_eval_style=True)
    print("Finished dataset preparation.")
    return dataset


def get_index_dataset(dataset: datasets.Dataset):
    return QuestionTableIndexDataset(dataset)


def linearize_table_with_separators(table_data, col_sep='#', row_sep='\n') -> str:
    #header_row = [
    #    header.replace('\n', ' ').strip() + col_sep
    #    if h < len(table_data['header']) - 1  # last column does not need column seperator
    #    else header.replace('\n', ' ').strip()
    #    for h, header in enumerate(table_data['header'])
    #    ]
    header_row = col_sep.join([header.replace('\n', ' ').strip() for header in table_data['data_dict']['header']])
    data_rows = [col_sep.join([cell.replace('\n', ' ').strip() for cell in row]) for row in table_data['data_dict']['rows']]
    return header_row + row_sep + row_sep.join(data_rows)


def get_table_dataset(absolute_table_dataset_path: str) -> datasets.Dataset:
    try:
        table_dataset = datasets.Dataset.load_from_disk(absolute_table_dataset_path)
    except FileNotFoundError as e:
        raise ValueError(f"A questions's path to it's table dataset could not be resolved!\n(path not found: {absolute_table_dataset_path})\n"
                          "This might happen if the system path changed or the corresponding table dataset was moved. "
                          "You can fix this by calling apply_table_dataset_path_changes (data_utils.py) with a mapping of paths that should be updated. "  # TODO implement path_map_function that updates the absolute paths
                          f"Use a path relative to the current working directory ({os.getcwd()})."
                          ) from e
    return table_dataset


def is_question_table_index_style_sample(question_data) -> bool:
    # QuestionTableIndexDataset style samples contain key 'data' as either datasets.Dataset or List[dict] type
    return (
        'table_idx' in question_data.keys() or  # if self.lm_eval_style is True table_idx will be in the samples' keys
        isinstance(question_data.get('data'), datasets.Dataset) or
        (isinstance(question_data.get('data'), list) and isinstance(question_data['data'][0], dict))
        )


def short_tabfact_sep_prompt(question_data, is_inference=False, question_only_format=True, cot=False, deepseek=False, col_sep=', ', row_sep='\n', base_prompt=PROMPT_TEMPLATE_TABLE_FIRST, table_index_path='tmp_table_index.pickle'):
    if is_question_table_index_style_sample(question_data):
        if 'table_dataset_path' in question_data.keys():
            table_data = [datasets.Dataset.load_from_disk(question_data['table_dataset_path'])[question_data['table_idx']]['table']]
            question_data = {'questions': [question_data['question']],
                             'answers': [question_data['answer']],
                             }
        else:
            # reformat data to be consistent with other conditions
            table_data = [question_data['data'][0]['table']]
            question_data = {'questions': [question_data['data'][0]['questions'][question_data['question_number']]],
                             'answers': [question_data['data'][0]['answers'][question_data['question_number']]],
                             }
    elif question_only_format:
        with open(table_index_path, 'rb') as f:
            table_index = pickle.load(f)
        absolute_table_dataset_path = question_data['table_dataset_path']
        table_dataset = get_table_dataset(absolute_table_dataset_path)
        # select table of current question
        if 'table' in table_dataset.column_names:
            # flatten fields (remove table field)
            table_dataset = table_dataset.map(lambda x: {key: value for key, value in x['table'].items()}, remove_columns='table', desc='Flatten table fields...')
        if table_index_path is not None:  # fast lookup of table
            table_idx = table_index.get(question_data['table'])
            if table_idx is None:
                raise KeyError(f"Table ID {question_data['table']} was not found in the table_index!")
            table_data = table_dataset.select([table_idx])
        else:
            table_data = table_dataset.filter(lambda example: example['table_id'] == question_data['table'])
            if len(table_data) != 1:
                for tab in table_data:
                    print(tab['data_dict']['header'])
                raise ValueError(f"Expected exactly one table to match given table_id {question_data['table']} but {len(table_data)} elements were found!")
    else:
        table_data = [question_data['table']]  # wrap in list to be consistent with question_only_format (Iterable of len() == 1)
    prompt_template = (
        base_prompt
        + ("Answer: {answer_text}" if not cot else "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n{answer_text}")
    )
    answer_text = question_data['answers'][0] if not cot else f"\\boxed{{{question_data['answers'][0]}}}"
    return prompt_template.format(
        table_text=linearize_table_with_separators(table_data[0],  # the [0] is because we want a single sample (dict) not a datasets.Dataset with len() == 1
                                                   col_sep=col_sep,
                                                   row_sep=row_sep
                                                   ),
        col_sep_repr=repr(col_sep),
        row_sep_repr=repr(row_sep),
        question_text=question_data['questions'][0],  # TODO singular key: question and no index
        answer_text=answer_text if not is_inference else ('' if not deepseek else "<think>\n")  # TODO singular key: answer and no index
        )


def short_tabfact_sep_prompt_inference(table_data):
    return short_tabfact_sep_prompt(table_data, is_inference=True)


def plain_single_answer(table_data):
    if is_question_table_index_style_sample(table_data):
        if 'answer' in table_data.keys():
            return table_data['answer']
        return table_data['data'][0]['answers'][table_data['question_number']]
    return table_data['answers'][0]
