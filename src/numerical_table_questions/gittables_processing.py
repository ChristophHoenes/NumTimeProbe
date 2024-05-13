import logging
import logging.config
import math
import warnings
from pathlib import Path, PurePath
from typing import Optional, Tuple, Dict

import numpy as np
import pyarrow.parquet as pq

from numerical_table_questions.data_caching import caching, save_version
from numerical_table_questions.data_synthesis import Table


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


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


def load_parquet_table(filepath: str, table_corpus_dir: Optional[str] = '/home/mamba/.cache/GitTables'):
    """ table_corpus_dir points to GitTables base dir location so only relative position within Gittables needs to be provided.
        If table_corpus_dir is None filepath must contain the full path to the file.
    """
    if table_corpus_dir is not None:
        complete_path = table_corpus_dir + '/' + filepath
    else:
        complete_path = filepath
    table_name = infer_table_name_from_path(complete_path)
    parquet_table = pq.read_table(complete_path)
    df = parquet_table.to_pandas()
    df = df.astype(str)  # convert all values into str
    df2dict = df.to_dict('tight')
    data_dict = {'header': df2dict['columns'], 'name': table_name, 'rows': df2dict['data']}
    return data_dict


def create_git_tables_dataset(data_dir_path: str = '/home/mamba/.cache/GitTables',
                              dataset_name='subset_10',
                              use_cache: bool = True,
                              cache_path: str = '/home/mamba/.cache',
                              topic_cutoff: Optional[int] = 10,  # limits data size to first x tables per topic (x*num_topics)
                              splits: Tuple[float, ...] = (0.8, 0.1, 0.1),
                              allow_modified_structure: bool = False,
                              save: bool = True,
                              ) -> Dict[str, Table]:
    data_dir_path_obj = Path(data_dir_path)
    cache_file_name = f"gittables_{dataset_name}_<SPLIT_NAME>_tables"
    if not data_dir_path_obj.exists():
        raise NotADirectoryError("The provided data_dir_path was not found!")
    if sum(splits) != 1.0:
        raise ValueError(f"The proporions of all splits must add up to exactly one but add up to {sum(splits)} instead!")
    match len(splits):
        case 0 | 1: split_tables = {'all': []},
        case 2: split_tables = {'train': [], 'test': []}
        case 3: split_tables = {'train': [], 'validation': [], 'test': []}
        case _: split_tables = {'train': [], 'validation': [], 'test': []} | {f'split_{3+idx}': [] for idx in range(len(splits)-3)}
    split_name_map = {i: split_name for i, split_name in enumerate(split_tables.keys())}
    no_splits_found = False  # only relevant if use_cache=True (caching is looking for splits)
    if use_cache:
        loaded_split_tables = {split_name: caching(cache_file_name.replace('<SPLIT_NAME>', 'split_name'), cache_path=cache_path)
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
        for child_path in data_dir_path_obj.iterdir():
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
            num_tables_topic = len(file_names)
            table_budget = min(num_tables_topic, (topic_cutoff or float('inf')))
            if table_budget < len(splits):
                warnings.warn(f"Not enough tables available in topic {root_dir.name} to form {len(splits)} splits!"
                              "Skipping this topic..."
                              )
                # TODO better add all tables to majority split (and maybe compensate afterwards? depending on how many times this happens)
                # or do stochastic assignment proportional to split sizes
                continue
            split_sizes = [math.floor((table_budget - len(splits)) * split_percentage) + 1 for split_percentage in splits]
            num_remaining_tables = table_budget - sum(split_sizes)
            for _ in range(num_remaining_tables):
                winner_idx = np.random.choice(len(splits), p=splits, replace=True)
                split_sizes[winner_idx] += 1
            split_counter = {idx: 0 for idx in range(len(splits))}
            filled_splits = 0
            # TODO maybe sample start indixes per split to have diverse table types (take care of in bounds); But are tables iterated in order?
            # calculate how many tables of this topic should be used and how to distribute them across splits
            for f, file_name in enumerate(file_names):
                # stops early in every topic if not the entire dataset should be used
                if f == (topic_cutoff or float('inf')):
                    break
                # splits are filled with tables one after the other to increase the probability of having related tables in the same split
                # if one split has reached its pre-computed split_size move on to next split
                if split_counter[filled_splits] == split_sizes[filled_splits]:
                    filled_splits += 1
                split_name = split_name_map[filled_splits]
                # load parquet file and transform inti Table format
                data_dict = load_parquet_table(filepath=str(root_dir / file_name),  # already contains path to GitTables base dir
                                               # passing None is important since otherwise default location of dataset is prepended
                                               # (but is already included in filepath -> twice -> invalid path)
                                               table_corpus_dir=None,
                                               )
                # TODO make tables unique (test if it is already the case)
                tab = Table(data_dict,
                            source_name=dataset_name,
                            source_split=split_name,  # TODO think if splitname is meaningful as gittables has no split
                            )
                split_tables[split_name].append(tab)  # append to appropriate split
                split_counter[filled_splits] += 1  # increment counter for current split
        if save:
            for split_name, split_table_list in split_tables.items():
                for table in split_table_list:
                    table.prepare_for_pickle()
                save_version(split_table_list, cache_path, cache_file_name.replace('<SPLIT_NAME>', split_name))
    return split_tables


def main():
    # load_parquet_table(filepath='data/NumTabQA/00-01.parquet', table_corpus_dir=None)
    git_tables = create_git_tables_dataset(use_cache=True)


if __name__ == "__main__":
    main()
