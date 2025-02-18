import copy
import errno
import glob
import os
import shutil
import tempfile
import warnings
from functools import partial
from dataclasses import asdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import datasets
import lightning as L
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.auto.tokenization_auto import AutoTokenizer

from numerical_table_questions.arguments import TrainingArgs, TokenizationArgs
from numerical_table_questions.sqlcoder_model import sqlcoder_prompt_template
from numerical_table_questions.utils.data_caching import caching, save_version
from numerical_table_questions.utils.data_utils import create_table_index
from numerical_table_questions.utils.model_utils import ModelTypeInfo
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset, table_collate
from numerical_table_questions.utils.model_utils import get_model_type_info
from numerical_table_questions.utils.tokenizer_utils import get_tokenizer, prepare_for_tokenizer, model_specific_tokenizing, post_tokenizing, restore_metadata
from numerical_table_questions.utils.dlib.frameworks.pytorch import (
    get_rank,
    main_process_first,
    set_torch_file_sharing_strategy_to_system,
)

if TYPE_CHECKING:
    from train import MiscArgs, TrainingArgs


class WrapCustomTupleDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.custom_tuple = kwargs.pop('custom_tuple', tuple())
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return _WrapCustomTupleDataLoaderIter(iter(self), self.custom_tuple)


class _WrapCustomTupleDataLoaderIter:
    """
        Class to wrap the output of next() of any Iterator
        (designed for Pytorch _BasicDataLoaderIter)
        into a custom tuple.
        The remaining functionality should stay the same as in the wrapped Iterator.
    """
    def __init__(self, wrapped_iter, custom_tuple=tuple()):
        if not isinstance(custom_tuple, tuple):
            raise ValueError(f"Argument 'custom_tuple' must be of type tuple not '{type(custom_tuple)}'!")
        self.wrapped_iter = wrapped_iter
        self.custom_tuple = custom_tuple

    def __iter__(self):
        return iter(self.wrapped_iter)

    def __next__(self):
        yielded_data = next(self.wrapped_iter)
        return tuple(yielded_data, *self.custom_tuple)

    def __len__(self):
        return len(self.wrapped_iter)


def path_from_components(table_corpus, dataset_name, split, model_name=None, data_dir: str = './data/NumTabQA/.cache') -> str:
    if model_name is None:
        # only filename of table question dataset (grouped by table; pre-tokenization) -> use with caching
        path = f"{table_corpus}_{split}_{dataset_name}"
    else:
        # teturn path of tokenited dataset
        path = Path(data_dir) / 'viable_tensors' / f"{table_corpus}_{dataset_name}_{model_name}_tokenized" / split
    return str(path)


# TODO consider moving within TableQADataModule -> less arguments as most contained in self but reuse outside of class needed?
def load_split_tensor(split: str, table_corpus: str, dataset_name: str, model_name: str,
                      data_dir: str = './data/NumTabQA/.cache', full_path=None, output_dict: bool = False):
    if full_path is None:
        full_path = path_from_components(table_corpus, dataset_name, split, model_name=model_name, data_dir=data_dir)
    data_dict = datasets.Dataset.load_from_disk(full_path).with_format('torch')

    if output_dict:
        # simply uses datasets.Dataset as input to dataloader -> a batch will be a dictionary of all dataset columns
        # this might have a small performance downside due to memory mapping <- TODO verify performance comparison
        return data_dict

    # whole dataset needs to fit into memory, takes longer to initially prepare dataloaders
    # but might save some time during iterating over the batches
    else:
        # apply attribute filter specified in model info
        model_typ_info = get_model_type_info(model_name)
        if model_typ_info.filter_data_attributes is not None:
            inputs_tensors = [data_dict[col] for col in data_dict.column_names
                              if col in model_typ_info.filter_data_attributes
                              or len(model_typ_info.filter_data_attributes) == 0
                              ]
            if len(inputs_tensors) == 0:
                raise ValueError(f"None of the specified columns {model_typ_info.filter_data_attributes} was found in the data! "
                                 f"Check the filters specified in ModelTypeInfo of '{model_name}' or the naming convention in the data dict."
                                 )
            elif len(inputs_tensors) < len(model_typ_info.filter_data_attributes):
                warnings.warn(f"Some of the specified filters {model_typ_info.filter_data_attributes} were not found in the data dict and stay empty! "
                              "Tensor indices might be different from expected position."
                              )
        else:
            # if no explicit column filter is specified in model info ensure only inputs with the correct size are forwarded
            # to the dataloader (requires all tensors to have same length)
            main_input_col_name = 'input_ids'
            tensor_length = data_dict[main_input_col_name].size()[0]
            inputs_tensors = [data_dict[col] for col in data_dict.column_names
                              if col != 'targets'
                              and isinstance(data_dict[col], torch.Tensor)
                              and data_dict[col].size()[0] == tensor_length
                              ]
        # all dataset columns (except 'targets') as in-memory tensor dataset (the dataloader should return batch as tuple)
        inputs_dataset = torch.utils.data.TensorDataset(*inputs_tensors)
        if 'targets' in data_dict.column_names:
            # dataloader should return batch as tuple of (inputs, targets)
            # where inputs is a tuple of all dataset columns except 'targets' and targets is the dataset 'targets' column
            return torch.utils.data.StackDataset(inputs_dataset, data_dict['targets'])
        return inputs_dataset


class TableQADataModule(L.LightningDataModule):
    def __init__(self,
                 model_specs,
                 table_corpus='wikitables',
                 dataset_name='basic_dataset',
                 train_batch_size: int = 32,
                 eval_batch_size: int = 64,
                 tokenizing_args=None,
                 lazy_data_processing: bool = True,
                 is_batch_dict: bool = True,
                 validation_cutoff=2048,
                 data_dir: str = './data/NumTabQA/.cache',
                 overwrite_cache: bool = False,
                 num_dataloader_workers: int = 0,
                 too_many_open_files_fix: bool = False,
                 ):
        super().__init__()
        self.model_specs = model_specs
        # TODO test if model name is known else raise NotImplemented error
        self.model_name = self.model_specs.model_name_or_path
        self.table_corpus = table_corpus
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.lazy_data_processing = lazy_data_processing
        # TODO maybe implement tensor batch conversion (but maybe pointless as no performance gain when done on the fly?)
        if not is_batch_dict and lazy_data_processing:
            warnings.warn("For lazy_data_processing batch will always be a dict! is_batch_dict=True was set automatically.")
            is_batch_dict = True
        self.is_batch_dict = is_batch_dict
        self.validation_cutoff = validation_cutoff
        self.tokenizing_args = asdict(tokenizing_args) if tokenizing_args is not None else dict()
        self.tokenizer = get_tokenizer(self.model_name, **self.tokenizing_args)
        self.max_num_tokens = self.tokenizing_args.get('max_length') or 1024
        self.data_dir = data_dir
        self.overwrite_cache = overwrite_cache
        self.splits = dict()
        self.too_many_open_files_fix = too_many_open_files_fix
        self.num_dataloader_workers = num_dataloader_workers
        self.data_loader_args = dict(
            num_workers=self.num_dataloader_workers,
            persistent_workers=(True if self.num_dataloader_workers > 0 else False),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.too_many_open_files_fix
            else None,
        )

    def process_sequences_from_intermediate(self, intermediate_dict: dict, huggingface_base_dir: str, split: str):

        # model specific custom post-tokenization processing (truncation, padding, filtering tokenizer outputs, adding additional fields, etc.)
        processed_sequences = post_tokenizing(intermediate_dict,
                                              self.tokenizing_args,
                                              self.max_num_tokens,
                                              self.model_specs.pad_token_id,
                                              self.model_specs.mask_token_id,
                                              self.model_name,
                                              )
        # save fully processed dataset
        # TODO think about what to do with other files that might already be in this directory but have a different
        # name (e.g. pickle or different shard numbers) so they do not get overwritten
        # (e.g delete, move to subfolder with previous versions)
        datasets.Dataset.from_dict(processed_sequences).save_to_disk(
            self.data_dir
            + '/viable_tensors/'
            + huggingface_base_dir
            + f"/{split}"
        )
        # save as metadata (in extra text file) the length of the dataset after post_tokenizing
        num_samples_after_filtering = len(processed_sequences['input_ids'])
        with (Path(self.data_dir) / 'viable_tensors' / huggingface_base_dir / split / 'custom_metadata.txt').open('a+') as f:
            f.write(f"{datetime.now().strftime('%y%m%d_%H%M_%S_%f')} num_rows {num_samples_after_filtering}\n")

    def prepare_data(self):
        # download not needed as locally on disk from data_synthesis
        if self.lazy_data_processing:
            return  # no preparation needed if it is doene on the fly during data loading
        # but once published download from huggingface datasets
        for split in ['train', 'validation', 'test']:
            logger.info(f"Preparing split '{split}'")
            # path definitions to check for saved files
            huggingface_base_dir = f"{self.table_corpus}_{self.dataset_name}_{self.model_name}_tokenized"
            final_processing_path = Path(self.data_dir) / 'viable_tensors' / huggingface_base_dir / split
            intermediate_processing_path = Path(self.data_dir) / 'full_dict' / huggingface_base_dir / split

            if final_processing_path.exists() and not self.overwrite_cache:
                # load fully processed tensor dataset to ensure no error occurs
                logger.info(f"Found processed 'viable_tensors' file for split '{split}'. Loading...")
                data_split = datasets.load_from_disk(final_processing_path)
            elif intermediate_processing_path.exists() and not self.overwrite_cache:
                logger.info(f"Found intermediate 'full_dict' file for split '{split}'. Execute post-tokenization...")
                # load from intermediate step (all examples) and apply custom post-processing and filtering
                tokenized_dict = datasets.load_from_disk(intermediate_processing_path).with_format('torch')
                # convert to dict while keeping the tensor format
                tokenized_dict = {field: [tokenized_dict[i][field] for i in range(tokenized_dict.num_rows)]
                                  for field in tokenized_dict.column_names
                                  }

                # run model specific custom post-tokenization processing (e.g. truncation, padding, etc.) and save results
                self.process_sequences_from_intermediate(tokenized_dict, huggingface_base_dir, split)
            else:
                # load raw TableQuestionDataset and do full processing
                # TODO replace 'wikitables' with variable to allow for different source datasets
                base_filename = f"{self.table_corpus}_{split}_{self.dataset_name}"
                data_split = caching(base_filename, cache_path=self.data_dir)
                if data_split is None:
                    raise ValueError(f"No data split '{split}' found at {self.data_dir} for dataset "
                                     f"{self.dataset_name} based on table corpus {self.table_corpus}! "
                                     "Please download, or generate the requested dataset.")
                # always disable padding and truncation - apply configuration afterwards
                # except for special truncation strategies supported by the tokenizer
                tokenizing_args = self.tokenizing_args.copy()
                tokenizing_args.update({'padding': False,
                                        'truncation': self.tokenizing_args['truncation']
                                        if self.tokenizing_args['allow_custom_truncation'] else False
                                        }
                                       )
                # TODO memory mapped
                # transform input to format expected by tokenizer (only considering input and target fields)
                tokenizer_inputs = prepare_for_tokenizer(data_split, self.model_name, is_eval=(split == 'test'), **tokenizing_args)
                logger.info("Tokenize examples...")
                # run tokenization and return tokenized fields
                # from list of tokenizer input fields (list of dicts) to dict of tokenizer output fields (dict with lists of question samples as values)
                tokenized_dict = model_specific_tokenizing(self.tokenizer, tokenizer_inputs, self.model_name,
                                                           self.model_specs.pad_token_id, self.model_specs.mask_token_id,
                                                           **tokenizing_args,
                                                           )

                # add back the fields that did not go through tokenization
                restore_metadata(data_split, tokenized_dict)

                # save raw tokenizer outputs (sequences with variable length)
                datasets.Dataset.from_dict(tokenized_dict).save_to_disk(
                    self.data_dir
                    + '/full_dict/'
                    + huggingface_base_dir
                    + f"/{split}"
                )
                # save as metadata (in extra text file) the length of the dataset before post_tokenizing (e.g. before filtering too long sequences)
                num_samples_before_filtering = len(tokenized_dict.get('input_ids', []))
                with (Path(self.data_dir) / 'full_dict' / huggingface_base_dir / split / 'custom_metadata.txt').open('a+') as f:
                    f.write(f"{datetime.now().strftime('%y%m%d_%H%M_%S_%f')} num_rows {num_samples_before_filtering}\n")

                # run model specific custom post-tokenization processing (e.g. truncation, padding, etc.) and save results
                self.process_sequences_from_intermediate(tokenized_dict, huggingface_base_dir, split)

    def setup(self, stage: str):
        print('setup', stage)

        def check_dataset_type(split_name):
            if not self.is_batch_dict and not isinstance(self.splits[split_name], torch.utils.data.StackDataset):
                # TODO think of using TypeError instead
                warnings.warn(
                    f"Dataset should have type 'torch.utils.data.StackDataset' if there are targets but is of type '{type(self.splits[split_name])}'! "
                    "There should always be targets available check your dataset."
                )
                # raise TypeError(f"Dataset should have type 'torch.utils.data.StackDataset' but is of type '{type(self.splits[split_name])}'! "
                #                "Dataset should return a variable length tuple of model inputs and the targets as torch.Tensor.")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            if self.lazy_data_processing:
                self.splits['train'] = QuestionTableIndexDataset(path_from_components(self.table_corpus, self.dataset_name, 'train'), data_dir=self.data_dir)
                self.splits['validation'] = QuestionTableIndexDataset(path_from_components(self.table_corpus, self.dataset_name, 'validation'), data_dir=self.data_dir, cutoff=self.validation_cutoff)
            else:
                self.splits['train'] = load_split_tensor('train', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('train')
                self.splits['validation'] = load_split_tensor('validation', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('validation')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            if self.lazy_data_processing:
                self.splits['test'] = QuestionTableIndexDataset(path_from_components(self.table_corpus, self.dataset_name, 'test'), data_dir=self.data_dir)#, cutoff=20_000)  # TODO remove cutoff after debug
            else:
                self.splits['test'] = load_split_tensor('test', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('test')

        if stage == 'predict':
            if self.lazy_data_processing:
                self.splits['test'] = QuestionTableIndexDataset(path_from_components(self.table_corpus, self.dataset_name, self.model_name, 'test'), data_dir=self.data_dir)
            else:
                self.splits['test'] = load_split_tensor('test', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('test')

    def _get_dataloader(self, split_name: str, split_config: dict) -> DataLoader:
        # determine collate function for processing during data loading
        if self.lazy_data_processing:
            # pre-computing table index speeds up collating
            table_index = None
            if isinstance(self.splits[split_name], QuestionTableIndexDataset):
                dataset = self.splits[split_name].table_dataset
                if len(dataset) > 0 and isinstance(dataset[0].get('table'), str):
                    table_dataset_path = dataset[0].get('table_dataset_path')
                    if table_dataset_path is None:
                        raise ValueError(
                            "The provided dataset does not contain the table data but also no path to the table dataset was provided. "
                            "Make sure table_dataset_path is set. "
                            "This can also be set manually (e.g data_utils.apply_table_dataset_path_changes) if the table dataset was moved to a different location."
                            )
                    else:
                        table_dataset = datasets.Dataset.load_from_disk(table_dataset_path)
                        table_index = create_table_index(table_dataset)

            collate_fn = partial(
                table_collate,
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                tokenizing_args=self.tokenizing_args,
                pad_token_id=self.model_specs.pad_token_id,
                mask_token_id=self.model_specs.mask_token_id,
                truncation=self.tokenizing_args['truncation'],
                padding=self.tokenizing_args['padding'],
                is_eval=(split_name == 'test'),  # for testing no answer coordinates are needed (tapas)
                #table_index=table_index,
                )
        else:
            collate_fn = None
        # load default config from self and (partially) override with custom split config
        data_loader_args = copy.deepcopy(self.data_loader_args)
        data_loader_args.update(split_config)
        # ensure to always have a batch in the format Tuple[Tuple[inputs], Union[Optional[target], Tuple[targets]]]
        if isinstance(self.splits[split_name], torch.utils.data.TensorDataset):
            return WrapCustomTupleDataLoader(self.splits[split_name], custom_tuple=(None,), collate_fn=collate_fn, **data_loader_args)
        return DataLoader(self.splits[split_name], collate_fn=collate_fn, **data_loader_args)

    def train_dataloader(self):
        return self._get_dataloader(split_name='train', split_config=dict(batch_size=self.train_batch_size, shuffle=True))

    def val_dataloader(self):
        return self._get_dataloader(split_name='validation', split_config=dict(batch_size=self.eval_batch_size))

    def test_dataloader(self):
        return self._get_dataloader(split_name='test', split_config=dict(batch_size=self.eval_batch_size))

    def predict_dataloader(self):
        return self._get_dataloader(split_name='test', split_config=dict(batch_size=self.eval_batch_size))


def batch_samples_collate(index_dataset_samples):
    # get table and question from dataset
    prepared_batch = {'table_idx': [], 'question_number': [], 'question_id': [], 'tables': [], 'questions': [], 'answers': []}
    for sample in index_dataset_samples:
        data = sample['data'][0]  # data is always a list of one element
        question_number = sample['question_number']
        prepared_batch['table_idx'].append(sample['table_idx'])
        prepared_batch['question_number'].append(question_number)
        prepared_batch['question_id'].append(sample['question_id'])
        prepared_batch['tables'].append(data['table'])
        prepared_batch['questions'].append(data['questions'][question_number])
        prepared_batch['answers'].append(data['answers'][question_number])
    return prepared_batch


class SQLCoderDataModule(TableQADataModule):
    def prepare_data(self):
        # download not needed as locally on disk from data_synthesis
        if self.lazy_data_processing:
            return  # no preparation needed if it is doene on the fly during data loading
        else:
            raise NotImplementedError("Data pre-processing is not implemented for SQLCoderDataModule use lazy_data_processing.")

    def setup(self, stage: str):
        if self.lazy_data_processing:
            super().setup(stage)
        else:
            raise NotImplementedError("Data pre-processing is not implemented for SQLCoderDataModule use lazy_data_processing.")

    def _get_dataloader(self, split_name: str, split_config: dict) -> DataLoader:
        # determine collate function for processing during data loading
        if self.lazy_data_processing:
            # load default config from self and (partially) override with custom split config
            data_loader_args = copy.deepcopy(self.data_loader_args)
            data_loader_args.update(split_config)
            return DataLoader(self.splits[split_name], collate_fn=batch_samples_collate, **data_loader_args)
        else:
            raise NotImplementedError("Data pre-processing is not implemented for SQLCoderDataModule use lazy_data_processing.")


def create_sqlcoder_dataset(eval_args: Optional[TrainingArgs] = None,
                            tokenizer_args: Optional[TokenizationArgs] = None,
                            split: str = 'test',
                            model_type_info: Optional[ModelTypeInfo] = None,
                            save_path: Optional[str] = None
                            ) -> datasets.Dataset:
    """ Creates a dataset.Dataset version from the custom lightning data module,
        For more efficient use of huggingface pipelines for inference (as used in our SQLCoder model)
        a dataset should be passed to the pipeline instead of calling a new pipeline for every batch of a DataLoader.
    """
    # if no arguments were provided use default values
    if eval_args is None:
        eval_args = TrainingArgs()
    if tokenizer_args is None:
        tokenizer_args = TokenizationArgs()

    # Initialize the data module that is appropriate for the model
    dm = SQLCoderDataModule(model_type_info or ModelTypeInfo(eval_args.model_name_or_path),
                            table_corpus=eval_args.table_corpus_name,
                            dataset_name=eval_args.dataset_suffix,
                            train_batch_size=eval_args.batch_size_per_device,
                            eval_batch_size=eval_args.eval_batch_size_per_device,
                            lazy_data_processing=eval_args.lazy_data_processing,
                            is_batch_dict=eval_args.is_batch_dict,
                            data_dir=eval_args.data_dir,
                            tokenizing_args=tokenizer_args,
                            num_dataloader_workers=eval_args.workers,
                            )
    match split.lower():
        case 'test':
            dm.setup('test')
            dataloader = dm.test_dataloader()
        case 'train':
            dm.setup('fit')
            dataloader = dm.train_dataloader()
        case 'validation':
            dm.setup('validate')
            dataloader = dm.val_dataloader()
        case _:  # default to test
            raise ValueError(f"Invalid split: {split}")

    batch_datasets = []
    for batch in dataloader:
        batch_questions = []
        for sample_idx in range(len(batch['questions'])):
            batch_questions.append(sqlcoder_prompt_template(batch['questions'][sample_idx], batch['tables'][sample_idx]))
        batch['questions'] = batch_questions
        batch_datasets.append(datasets.Dataset.from_dict(batch))
    dataset = datasets.concatenate_datasets(batch_datasets)
    if save_path is not None:
        save_version(dataset, save_path)
    return dataset
