import os
from pathlib import PurePath

import datasets


def process_docs(dataset: datasets.Dataset):
    print(dataset.column_names)
    return dataset


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


def short_tabfact_sep_prompt(question_data, is_inference=False, question_only_format=True, cot=False, col_sep=', ', row_sep='\n'):
    if question_only_format:
        absolute_table_dataset_path = question_data['table_dataset_path']
        try:
            table_dataset = datasets.Dataset.load_from_disk(absolute_table_dataset_path)
        except FileNotFoundError as e:
            raise ValueError(f"A questions's path to it's table dataset could not be resolved!\n(path not found: {absolute_table_dataset_path})\n"
                             "This might happen if the system path changed or the corresponding table dataset was moved. "
                             "You can fix this by calling apply_table_dataset_path_changes (data_utils.py) with a mapping of paths that should be updated. "  # TODO implement path_map_function that updates the absolute paths
                             f"Use a path relative to the current working directory ({os.getcwd()})."
                             ) from e
        # select table of current question
        if 'table' in table_dataset.column_names:
            table_dataset = table_dataset.map(lambda x: {key: value for key, value in x['table'].items()}, remove_columns='table', desc='Flatten table fields...')
        table_data = table_dataset.filter(lambda example: example['table_id'] == question_data['table'])
        if len(table_data) != 1:
            for tab in table_data:
                print(tab['data_dict']['header'])
            raise ValueError(f"Expected exactly one table to match given table_id {question_data['table']} but {len(table_data)} elements were found!")
    else:
        table_data = [question_data['table']]  # wrap in list to be consistent with question_only_format (Iterable of len() == 1)
    prompt_template_short = (
        "Table:\n"
        "{table_text}\n\n"
        "Note: {col_sep_repr} is the delimiter between columns; {row_sep_repr} is the delimiter between rows.\n\n"
        'Question: "{question_text}"\n\n'
        "Answer the question based on the table.|||\n\n"
        + ("Answer: {answer_text}" if not cot else "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\\boxed{{{answer_text}}}")
    )
    return prompt_template_short.format(
        table_text=linearize_table_with_separators(table_data[0],  # the [0] is because we want a single sample (dict) not a datasets.Dataset with len() == 1
                                                   col_sep=col_sep,
                                                   row_sep=row_sep
                                                   ),
        col_sep_repr=repr(col_sep),
        row_sep_repr=repr(row_sep),
        question_text=question_data['questions'][0],  # TODO singular key: question and no index
        answer_text=question_data['answers'][0] if not is_inference else ''  # TODO singular key: answer and no index
        )


def short_tabfact_sep_prompt_inference(table_data):
    return short_tabfact_sep_prompt(table_data, is_inference=True)


def plain_single_answer(table_data):
    return table_data['answers'][0]


def extract_from_pattern(table_data):
    return table_data['answers'][0]
