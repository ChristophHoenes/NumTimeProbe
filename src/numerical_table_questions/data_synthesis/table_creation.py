from __future__ import annotations

import hashlib
import logging
import logging.config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import PurePath
from typing import List, Dict, Optional, Union

import datasets
from dargparser import dargparse

from numerical_table_questions.arguments import DataProcessingArgs
from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.data_synthesis.table import Table, name_id_mapping


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


# change scope to load hf dataset
def load_hf_table_dataset(base_dataset_name: str = 'wikitablequestions',
                          base_dataset_split: str = 'test',
                          num_tables: Optional[int] = None,
                          ) -> datasets.Dataset:
    logger.info("Loading %s's first %s %s split samples",
                base_dataset_name,
                str(num_tables or 'all'),
                base_dataset_split
                )
    dataset_slice = '[:{num_tables}]' if num_tables is not None else ''
    dataset = datasets.load_dataset(
        base_dataset_name,
        split=f'{base_dataset_split}{dataset_slice}',
        )
    return dataset


def full_table_hash(table) -> str:
    return hashlib.sha256(str.encode(str(table))).hexdigest()


def header_len_table_hash(table: Union[list, dict], return_length: bool = False) -> Union[str, Tuple[str, int]]:
    if isinstance(table, list):
        # define header as first row (or element in list)
        header = table[0]
        length = len(table)
    elif isinstance(table, dict):
        header = table.get('header')
        if header is None:
            raise ValueError("For this hash function table needs to be a list or a dict which contains the key 'header' with a non-empty value!")
        rows = table.get('rows')
        if rows is None:
            raise ValueError("For this hash function table needs to be a list or a dict which contains the key 'rows' with a list of table rows as value!")
        length = len(rows)
    if return_length:
        return hashlib.sha256(str.encode(str(header) + str(length))).hexdigest(), length
    return hashlib.sha256(str.encode(str(header) + str(length))).hexdigest()


def sample_table_hash(table: Union[list, dict], sample_ids=(0, 5, 100)) -> str:
    header_len_hash, length = header_len_table_hash(table, return_length=True)
    if isinstance(table, list):
        samples = [str(table[sample_id % length]) for sample_id in sample_ids]
    else:
        samples = [str(table['rows'][sample_id % length]) for sample_id in sample_ids]
    sample_hash = hashlib.sha256(str.encode(';'.join(samples))).hexdigest()
    return hashlib.sha256(str.encode(header_len_hash + sample_hash)).hexdigest()


def add_table_hashes(dataset: datasets.Dataset, table_field='table', hash_func=full_table_hash, num_proc: int = 12) -> datasets.Dataset:
    return dataset.map(
        lambda x: {'table_id': hash_func(x[table_field])},
        num_proc=num_proc,
        desc="Creating table hashes as table_id..."
        )


def deduplicate_tables(dataset: datasets.Dataset) -> datasets.Dataset:
    if len(dataset) == len(set(dataset['table_id'])):
        logger.info("deduplicate_tables: Dataset already duplicate free.")
        # CAUTION: if dataset is duplicate-free same object is returned if not new dataset is returned
        # -> different delete cache behavior
        return dataset
    table_counter = {}
    select_indices = []
    for idx, table_id in enumerate(dataset['table_id']):
        if table_counter.get(table_id) is None:
            table_counter[table_id] = 1
            select_indices.append(idx)
        else:
            table_counter[table_id] += 1
    return dataset.select(select_indices)


def create_table_dataset(base_dataset: datasets.Dataset,
                         cache_path: str = './data/NumTabQA/.cache',  # TODO move to config
                         save: bool = True,
                         num_proc: int = 12,
                         ) -> datasets.Dataset:
    _required_fields = ('table', 'dataset_name', 'dataset_split')
    _is_requred_fields_present_dict = {field: field in base_dataset.column_names for field in _required_fields}
    if not all(_is_requred_fields_present_dict.values()):
        missing_fields = [field for field in _required_fields if not _is_requred_fields_present_dict[field]]
        raise ValueError(f"The base_dataset requires at least the following fields: {_required_fields}. "
                         f"The field(s) {','.join([missing_fields])} was/were not found. Please process the dataset accordingly."
                         )
    cache_file_name = f"{base_dataset[0]['dataset_name']}_{base_dataset[0]['dataset_split']}_tables"
    # remove all irrelevant columns
    base_dataset = base_dataset.map(lambda _: {},
                                    remove_columns=[col for col in base_dataset.column_names
                                                    if col not in _required_fields or col == 'table_id']
                                    )
    # CAUTION: depending on selected hash function does not test exact table equality of all values but only a proxy
    if 'table_id' not in base_dataset.column_names:
        dataset = add_table_hashes(base_dataset)
    # filters duplicate tables (near duplicates might be overridden depending on hash function used for table_id)
    dataset = deduplicate_tables(dataset)
    # generate custom table object
    dataset = dataset.map(lambda x: {'table': Table(x['table'], source_name=x['dataset_name'], source_split=x['dataset_split']).to_state_dict()},
                          num_proc=num_proc,
                          desc="Creating custom table objects..."
                          )
    if save:
        save_version(dataset, cache_path, cache_file_name)
    return dataset


def table_dataset_to_dict(dataset: datasets.Dataset) -> Dict[str, Table]:
    unique_tables = {}
    for table in dataset:
        unique_tables[table['table_id']] = Table.from_state_dict(table)
    return unique_tables


def remove_duplicate_qa_pairs(data_sample):
    """ Removes duplicate QA pairs within the table batch.
        This should only be necessary for old datasets.
        In newer versions duplicats should already have been removed during dataset synthesis.
    """
    unique_questions, unique_answers = list(zip(
        *set(zip(data_sample['questions'], data_sample['answers']))
        ))
    return {'questions': unique_questions, 'answers': unique_answers}


def gittables_main():
    table_dataset = load_table_dataset(table_corpus='gittables_subset_10', split='train', cache_path='/home/mamba/.cache')
    # run column_name deduplication (since code changed since table dataset creation)
    for table in table_dataset:
        table.column_names = tuple(table.deduplicate_column_names(table.column_names))
        table._col2idx, table._idx2col = name_id_mapping(table.column_names, both_ways=True)
    return table_dataset


def add_dataset_name_and_split(dataset: datasets.Dataset, dataset_name: str, dataset_split: str) -> datasets.Dataset:
    return dataset.map(lambda x: {'dataset_name': dataset_name, 'dataset_split': dataset_split}, desc="Adding dataset name and split...")


def main(hf_base_dataset: str, splits=('test', 'train', 'validation'), cache_path: str = './data/NumTabQA/.cache', custom_prepare_func=None):
    # create new version of table dataset
    for split in splits:
        logger.info(f"Creating {hf_base_dataset} {split} split table dataset...")
        # load base dataset from huggingface
        base_dataset = load_hf_table_dataset(base_dataset_name=hf_base_dataset, base_dataset_split=split)
        # transform structure to conform with expected input required keys
        base_dataset_prepared = add_dataset_name_and_split(base_dataset, dataset_name=hf_base_dataset, dataset_split=split)
        if custom_prepare_func is not None:
            base_dataset_prepared = custom_prepare_func(base_dataset_prepared)
        # use custom logic (deduplication, custom table object creation, etc.) to create table dataset
        table_dataset = create_table_dataset(base_dataset=base_dataset_prepared, cache_path=cache_path, save=True)
        logger.info(f"Finished creating dataset. Results saved at {table_dataset.cache_files[0]['filename']} (and the potentially following arrow files).")


if __name__ == "__main__":
    args = dargparse(DataProcessingArgs)
    main(hf_base_dataset=args.table_corpus, splits=args.splits, cache_path=args.data_dir)
