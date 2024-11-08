from __future__ import annotations

import logging
import logging.config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import PurePath
from typing import List, Dict, Optional, Union

import datasets

from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.data_synthesis.table import Table, deduplicate_column_names, name_id_mapping


log_file_init_path = str(PurePath(__file__).parent.parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def load_table_dataset(table_corpus: str = 'wikitables',
                       split: Optional[str] = None,
                       cache_path: str = './data/NumTabQA/.cache',
                       memory_mapped: bool = True,
                       ) -> Optional[Union[List[Table], datasets.Dataset]]:
    cache_file_name = f"{table_corpus}_{split or 'all'}_tables"
    tables = caching(cache_file_name, cache_path=cache_path)
    if tables is not None and not memory_mapped:
        # restore original format in-memory by loading from state dict
        tables = [Table.from_state_dict(table_data) for table_data in tables]
    return tables


def create_table_dataset(base_dataset_name: str = 'wikitablequestions',
                         base_dataset_split: str = 'test',
                         num_tables: Optional[int] = None,
                         use_cache: bool = True,
                         cache_path: str = './data/NumTabQA/.cache',
                         save: bool = True,
                         ) -> Dict[str, Table]:
    cache_file_name = f'{base_dataset_name}_{base_dataset_split}_tables'
    if use_cache:
        tables = load_table_dataset(base_dataset_name, base_dataset_split, cache_path)
    if not use_cache or tables is None:
        logger.info("Loading %s's first %s %s split samples",
                    base_dataset_name, str(num_tables or 'all'), base_dataset_split)
        dataset_slice = '[:{num_tables}]' if num_tables is not None else ''
        dataset = datasets.load_dataset(
            base_dataset_name,
            split=f'{base_dataset_split}{dataset_slice}',
            )
        logger.info("Processing first %s tables of the test set...", str(num_tables or 'all'))
        # generate table object for every question in the source data split and use dict
        # to get a unique set of tables, by using _table_id as key near duplicates are overridden
        # Caution: _table_id does not test exact table equality of all values but only a proxy
        unique_tables = {}
        for i in range(len(dataset)):
            table = Table(dataset[i]['table'],
                          source_name=base_dataset_name,
                          source_split=base_dataset_split,
                          )
            unique_tables[table._table_id] = table
        tables = list(unique_tables.values())

        if save:
            for table in tables:
                table.prepare_for_pickle()
            save_version(tables, cache_path, cache_file_name)
    return tables


def remove_duplicate_qa_pairs(data_sample):
    """ Removes duplicate QA pairs within the table batch.
        This should only be necessary for old datasets.
        In newer versions duplicats should already have been removed during dataset synthesis.
    """
    unique_questions, unique_answers = list(zip(
        *set(zip(data_sample['questions'], data_sample['answers']))
        ))
    return {'questions': unique_questions, 'answers': unique_answers}


def main():
    #table_dataset = create_table_dataset(base_dataset_split='validation', use_cache=False)
    #return create_basic_table_question_dataset(table_dataset, name='count_wikitables_validation', use_cache=False)
    #create_basic_postprocessed_versions()
    table_dataset = load_table_dataset(table_corpus='gittables_subset_10', split='train', cache_path='/home/mamba/.cache')
    # run column_name deduplication (since code changed since table dataset creation)
    for table in table_dataset:
        table.column_names = tuple(deduplicate_column_names(table.column_names))
        table._col2idx, table._idx2col = name_id_mapping(table.column_names, both_ways=True)
    return table_dataset


if __name__ == "__main__":
    main()
