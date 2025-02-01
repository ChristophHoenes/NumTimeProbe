import logging
import logging.config
import math
import random
import re
import warnings
from pathlib import Path, PurePath
from typing import Optional, List, Tuple, Dict

import datasets
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from numerical_table_questions.utils.data_caching import caching, save_version
from numerical_table_questions.data_synthesis.table import Table, deduplicate_column_names
from numerical_table_questions.data_synthesis.template_creation import apply_quality_filters
from numerical_table_questions.data_utils import infer_is_multi_answer_posthoc


log_file_init_path = str(PurePath(__file__).parent.parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# from https://discuss.python.org/t/structural-pattern-matching-should-permit-regex-string-matches/22700/8 for easy match syntax
class REqual(str):
    "Override str.__eq__ to match a regex pattern."
    def __eq__(self, pattern):
        return re.fullmatch(pattern, self)


def infer_table_name_from_path(filepath: str):
    filepath_obj = Path(filepath)
    if not filepath_obj.is_file():
        raise FileNotFoundError("The provided filepath does not exist!")
    return filepath_obj.parent.name + '/' + filepath_obj.name


"""
def infer_table_name_from_path(filepath:str, table_corpus_dir: Optional[str] = 'data/NumTabQA/raw/GitTables'):
    filepath_obj = Path(filepath)
    if not filepath_obj.is_file():
        raise FileNotFoundError("The provided filepath does not exist!")
    file_name = filepath_obj.name
    dataset_prefix = ''
    if table_corpus_dir is None:
        dataset_prefix = filepath_obj.parent or ''
    else:
        dataset_dir_obj = Path(table_corpus_dir)
        if not dataset_dir_obj.exists():
            raise NotADirectoryError("The provided table_corpus_dir does not exist.")
        dataset_prefix = dataset_dir_obj.name
    return dataset_prefix + '/' + file_name
"""


def infer_data_type(data: str) -> str:
    match REqual(data.strip()):
        case r'[+-]?\d+': return 'int'
        case r'^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$': return 'float'
        case r'.*\d.*': return 'num_text'
        case _: return 'text'


def load_parquet_table(filepath: str, table_corpus_dir: Optional[str] = '/home/mamba/.cache/GitTables'):
    """ table_corpus_dir points to GitTables base dir location so only relative position within Gittables needs to be provided.
        If table_corpus_dir is None filepath must contain the full path to the file.
    """
    if table_corpus_dir is not None:
        complete_path = table_corpus_dir + '/' + filepath
    else:
        complete_path = filepath
    table_name = infer_table_name_from_path(complete_path)
    # read_table fails when there are duplicate column names which occurs in GitTables (e.g in 21th dir 6850th file column webdesign)
    # --> use workaround with ParquetFile (see issue: https://github.com/apache/arrow/issues/32660)
    #parquet_table = pq.read_table(complete_path)
    parquet_table = pq.ParquetFile(complete_path).read()
    try:
        df = parquet_table.to_pandas()
    except ValueError:
        parquet_table = parquet_table.rename_columns(deduplicate_column_names(parquet_table.column_names))
        df = parquet_table.to_pandas()
    df = df.astype(str)  # convert all values into str

    def is_header_data_row(dataframe: pd.DataFrame, num_samples: int = 5, len_diff_threshold: float = 0.1) -> bool:
        header_data_types = [infer_data_type(col) for col in dataframe.columns]
        # TODO more accurate would be majority of sample of rows or all rows
        row_samples = dataframe.sample(n=min(num_samples, len(dataframe)), replace=False)
        row_sample_data_types = [[infer_data_type(row.iloc[col_idx]) for r, row in row_samples.iterrows()] for col_idx in range(row_samples.shape[1])]
        # get most frequent type for each column
        row_majority_types = [max(set(row_types), key=row_types.count) for row_types in row_sample_data_types]

        data_like_types = ['float', 'int', 'num_text']
        # at least one unusual data-like value
        is_header_data_like = any([header_type in data_like_types for header_type in header_data_types])
        is_header_like_row = all([header_type == row_type for header_type, row_type in zip(header_data_types, row_majority_types)])
        if (is_header_data_like and is_header_like_row):
            return True
        elif not is_header_data_like and is_header_like_row:
            # usually when all columns are text -> will be filtered anyways but for correctness sake
            header_lens = [len(col) for col in dataframe.columns]
            row_sample_average_len = [row_samples.iloc[:, col_idx].apply(len).mean() for col_idx in range(row_samples.shape[1])]
            relative_len_diff = [(header_len-col_len) / max(col_len, header_len) for header_len, col_len in zip(header_lens, row_sample_average_len)]
            highest_len_difference = max([abs(diff) for diff in relative_len_diff])
            if highest_len_difference > len_diff_threshold:
                return False  # header lengths are different enough -> no data row
            else:
                return True  # header lengths are similar -> data row
        else:
            return False

    if is_header_data_row(df):
        header_row = df.columns
        df.iloc[0] = header_row
        df.columns = [f'col_{i}' for i in range(df.shape[1])]
    df2dict = df.to_dict('tight')
    data_dict = {'header': df2dict['columns'], 'name': table_name, 'rows': df2dict['data']}
    return data_dict


def create_git_tables_dataset(data_dir_path: str = '/home/mamba/.cache/GitTables',
                              dataset_name='full',
                              use_cache: bool = True,
                              cache_path: str = '/home/mamba/.cache',
                              topic_cutoff: Optional[int] = None,  # limits data size to first x tables per topic (x*num_topics)
                              splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
                              allow_modified_structure: bool = False,
                              group_similar_names: bool = False,
                              save: bool = True,
                              ) -> Dict[str, Table]:
    data_dir_path_obj = Path(data_dir_path)
    cache_file_name = f"gittables_{dataset_name}_<SPLIT_NAME>_tables"
    if not data_dir_path_obj.exists():
        raise NotADirectoryError("The provided data_dir_path was not found!")
    if sum(splits) != 1.0:
        raise ValueError(f"The proporions of all splits must add up to exactly one but add up to {sum(splits)} instead!")
    match len(splits):
        case 0 | 1: split_tables = {'all': []}
        case 2: split_tables = {'train': [], 'test': []}
        case 3: split_tables = {'train': [], 'validation': [], 'test': []}
        case _: split_tables = {'train': [], 'validation': [], 'test': []} | {f'split_{3+idx}': [] for idx in range(len(splits)-3)}
    split_name_map = {i: split_name for i, split_name in enumerate(split_tables.keys())}
    split_name_2_id = {split_name: i for i, split_name in enumerate(split_tables.keys())}
    no_splits_found = False  # only relevant if use_cache=True (caching is looking for splits)
    if use_cache:
        loaded_split_tables = {split_name: caching(cache_file_name.replace('<SPLIT_NAME>', split_name), cache_path=cache_path)
                               for split_name in split_name_map.values()
                               }
        splits_found = [tables is not None for tables in loaded_split_tables.values()]
        if all(splits_found):
            # restore original format by loading from state dict
            loaded_split_tables = {split_name: [Table.from_state_dict(table_data)
                                                for table_data in tables
                                                ]
                                   for split_name, tables in loaded_split_tables.items()
                                   }
            return loaded_split_tables
        elif any(splits_found):  # some but not all splits found
            raise ValueError("At leats one split was found while others are missing! "
                             "All splits must be created together to ensure integrety. "
                             "Check your 'splits' argument is consistent with your data or delete partial splits and recompute.")
        else:
            no_splits_found = True
            warnings.warn("No splits were found! They will be (re)computed.")
    if not use_cache or no_splits_found:
        logger.info("Loading GitTable's first %s tables per topic", str(topic_cutoff or 'all'))
        # gittables are grouped by subdirs that represent search terms (topics) for the tables
        """ Path.walk() requires python 3.12 but currently blocked at 3.11.9 by other dependencies -> use Path.iterdir() instead for now
        for (root_dir, subdir_names, file_names) in data_dir_path_obj.walk():
            if root_dir == data_dir_path_obj and len(file_names) > 0 and not allow_modified_structure:
                raise FileExistsError("Found a file at root level of GitTables dataset which is not expected in the original format! "
                                      "If you modified the dataset structure and want to supress this error pass allow_modified_structure=True.")
            if root_dir != data_dir_path_obj and len(subdir_names) > 0 and not allow_modified_structure:
                raise IsADirectoryError("Found a subdirectory in non-root-level directory of GitTables dataset which is not expected in the original format! "
                                        "If you modified the dataset structure and want to supress this error pass allow_modified_structure=True.")
        """
        # BEGIN ALTERNATIVE for Path.walk() implementation
        # TODO dynamically calculate total also considering cutoff
        total_files_progress_bar = tqdm(total=963798, position=0, desc='Percentage all GitTable files:')
        for child_path in tqdm(data_dir_path_obj.iterdir(), desc='Topic directories:'):
            if child_path.is_file() and not allow_modified_structure:
                raise FileExistsError("Found a file at root level of GitTables dataset which is not expected in the original format! "
                                      "If you modified the dataset structure and want to supress this error pass allow_modified_structure=True.")
            else:
                file_names = []
                root_dir = child_path  # translation to variable name in Path.walk() implementation
                for grandchild_path in child_path.iterdir():
                    if grandchild_path.is_dir() and not allow_modified_structure:
                        raise IsADirectoryError("Found a subdirectory in non-root-level directory of GitTables dataset which is not expected in the original format! "
                                                "If you modified the dataset structure and want to supress this error pass allow_modified_structure=True."
                                                )
                    elif grandchild_path.is_file():
                        file_names.append(grandchild_path.name)
            # END ALTERNATIVE for Path.walk() implementation
            if group_similar_names:
                file_names = group_name_neighbours(file_names)
            num_tables_topic = len(file_names)
            table_budget = min(num_tables_topic, (topic_cutoff or float('inf')))
            if table_budget < len(splits):
                warnings.warn(f"Not enough tables available in topic {root_dir.name} to form {len(splits)} splits!"
                              #"Skipping this topic..."
                              "Distributing with the following priority train, test, validation, other splits..."
                              )
                #continue  # skip topic
                # TODO better add all tables to majority split (and maybe compensate afterwards? depending on how many times this happens)
                # or do stochastic assignment proportional to split sizes
                num_remaining_tables = table_budget
                split_sizes = [0] * len(splits)
                # all non-prioritized splits (all but train and test)
                remaining_split_ids = list(set(range(len(splits))) - set([split_name_2_id['train'], split_name_2_id['test']]))
                remaining_split_percentages = [splits[split_id] for split_id in remaining_split_ids]
                for i in range(num_remaining_tables):
                    match i:
                        case 0: split_sizes[split_name_2_id['train']] += 1  # assign to train first
                        case 1: split_sizes[split_name_2_id['test']] += 1  # assign to test
                        case _: split_sizes[np.random.choice([remaining_split_ids], p=remaining_split_percentages)] += 1  # assign according to split sizes
            else:
                split_sizes = [math.floor((table_budget - len(splits)) * split_percentage) + 1 for split_percentage in splits]
                num_remaining_tables = table_budget - sum(split_sizes)
                for _ in range(num_remaining_tables):
                    winner_idx = np.random.choice(len(splits), p=splits, replace=True)
                    split_sizes[winner_idx] += 1
            split_counter = {idx: 0 for idx in range(len(splits))}
            filled_splits = 0
            # TODO maybe sample start indixes per split to have diverse table types (take care of in bounds); But are tables iterated in order?
            # calculate how many tables of this topic should be used and how to distribute them across splits
            for f, file_name in tqdm(enumerate(file_names), desc='Files in Topic:', total=len(file_names)):
                # stops early in every topic if not the entire dataset should be used
                if f == (topic_cutoff or float('inf')):
                    break
                # splits are filled with tables one after the other to increase the probability of having related tables in the same split
                # if one split has reached its pre-computed split_size move on to next split
                if split_counter[filled_splits] == split_sizes[filled_splits]:
                    filled_splits += 1
                split_name = split_name_map[filled_splits]
                # TODO make dataset of file names and then use map to load, process and write table sample by sample
                table_path = str(root_dir / file_name)  # already contains path to GitTables base dir
                split_tables[split_name].append({'filepath': table_path,  # append to appropriate split
                                                 'dataset_name': dataset_name,
                                                 'split_name': split_name,
                                                 }
                                                )
                split_counter[filled_splits] += 1  # increment counter for current split
            total_files_progress_bar.update(n=len(file_names))
        if save:
            for split_name, split_table_list in split_tables.items():
                save_version(split_table_list, cache_path, cache_file_name.replace('<SPLIT_NAME>', split_name))
    return split_tables


def load_and_process_parquet_to_table(sample):
    # load parquet file and transform inti Table format
    data_dict = load_parquet_table(filepath=sample['filepath'],
                                   # passing None is important since otherwise default location of dataset is prepended
                                   # (but is already included in filepath -> twice -> invalid path)
                                   table_corpus_dir=None,
                                   )
    # TODO make tables unique (test if it is already the case)
    tab = Table(data_dict,
                source_name=sample['dataset_name'],
                source_split=['split_name'],  # TODO think if splitname is meaningful as gittables has no split
                )
    return tab.to_state_dict()


# post-hoc column_name deduplication
def post_hoc_column_name_deduplication(sample):
    tab = Table.from_state_dict(sample)
    tab.column_names = tuple(
        deduplicate_column_names(tab._data_dict['header'])
        )
    return tab.to_state_dict()


def group_name_neighbours(file_names: List[str], num_group_samples: int = 1) -> List[str]:
    selected_names = []
    group_names = []
    first_group_name_parts = []
    for name in sorted(file_names):  # in alphabetical order (neighbours)
        name_parts = re.split('[^a-zA-Z0-9]', name)
        name_parts.pop()  # last one is always parquet ending
        # group together if any name parts have exact match
        if any([name_part == group_part for name_part, group_part in zip(name_parts, first_group_name_parts)]):
            group_names.append(name)
            continue
        # select/sample names from the current/old group
        chosen_group_names = random.sample(group_names, k=min(num_group_samples, len(group_names)))
        selected_names.extend(chosen_group_names)
        # start new group
        group_names = [name]
        first_group_name_parts = name_parts
    # sample for last group
    chosen_group_names = random.sample(group_names, k=min(num_group_samples, len(group_names)))
    selected_names.extend(chosen_group_names)
    return selected_names


# Not recommended
def group_name_structure(file_names: List[str], num_group_samples: int = 2) -> List[str]:
    def get_structure_code(input: str):
        match REqual(input):
            case r'\d+': return f'd{len(input)}'
            case r'[a-z]+': return 't'
            case r'[A-Z]+': return 'T'
            case r'[a-zA-Z]+': return 'tT'
            case _: return f'alpha_num{len(input)}'
    structure_groups = {}
    for name in file_names:
        name_parts = re.split('[^a-zA-Z0-9]', name)
        name_parts.pop()  # last one is always parquet ending
        special_characters = [char for char in re.split('[a-zA-Z0-9]', name) if char != '']
        special_characters[-1] = ''  # last one is always . before parquet ending -> overwrite with empty string instead of pop to make of same length as name_parts
        structure_code = ''.join([get_structure_code(name_part) + special_characters[n] for n, name_part in enumerate(name_parts)])
        if structure_groups.get(structure_code):
            structure_groups[structure_code].append(name)
        else:
            structure_groups[structure_code] = [name]
    selected_names = []
    for structure_code, group_names in structure_groups.items():
        # sample for last group
        chosen_group_names = random.sample(group_names, k=min(num_group_samples, len(group_names)))
        selected_names.extend(chosen_group_names)
    return selected_names


def subsample_groups():
    pass


def main():
    dataset_name = 'group_filtered'
    cache_path = '/home/mamba/.cache'

    # test manual parquet table load
    # load_parquet_table(filepath='data/NumTabQA/00-01.parquet', table_corpus_dir=None)

    # create filepaths splits dataset
    #git_tables = create_git_tables_dataset(dataset_name=dataset_name, cache_path=cache_path, use_cache=False, group_similar_names=True)
    #print(len(git_tables['test']))

    # load filepaths datasets and execute table processing
    train = caching(cache_file_name=f'gittables_{dataset_name}_validation_tables', cache_path=cache_path)
    train = train.map(load_and_process_parquet_to_table, num_proc=24, desc="Processing Tables...")
    save_version(train, cache_path, cache_file_name=f'gittables_{dataset_name}_validation_tables')


def quality_filtering(data_version_name='gittables_group_filtered_standard_templates_test', cache_path='/home/mamba/.cache/', num_proc: int = 12):
    latest_data = caching(data_version_name, cache_path=cache_path)
    latest_data = latest_data.map(infer_is_multi_answer_posthoc, num_proc=num_proc, desc="set is_multy_row post-hoc...")
    apply_quality_filters(latest_data,
                          remove_multi_answer=True,
                          single_row_agg_tolerances=(0.0,),
                          threshold=2,
                          save=True,
                          dataset_name=data_version_name,
                          num_proc=num_proc,
                          cache_path=cache_path,
                          )


if __name__ == "__main__":
    main()
