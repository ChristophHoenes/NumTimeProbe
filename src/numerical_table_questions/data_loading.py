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

from numerical_table_questions.data_caching import caching
from numerical_table_questions.data_utils import create_table_index
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset, table_collate
from numerical_table_questions.model_utils import get_model_type_info
from numerical_table_questions.tokenizer_utils import get_tokenizer, prepare_for_tokenizer, model_specific_tokenizing, post_tokenizing, restore_metadata
from numerical_table_questions.dlib.frameworks.pytorch import (
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


class LMDataModule(L.LightningDataModule):
    def __init__(
        self,
        training_args: "TrainingArgs",
        misc_args: "MiscArgs",
        mlm_probability=0.15,
        whole_word_masking=False,
    ):
        super().__init__()
        self.args = training_args
        self.misc_args = misc_args
        self.data_dir = (
            Path(training_args.data_dir) / training_args.language
            if training_args.language
            else Path(training_args.data_dir)
        )
        train_file, dev_file = (
            self.data_dir / self.args.train_file,
            self.data_dir / self.args.dev_file,
        )

        logger.debug(f"Train file path: {train_file} Dev file path: {dev_file}")

        self.train_file = str(train_file)
        self.dev_file = str(dev_file)
        self.mlm_probability = mlm_probability
        self.whole_word_masking = whole_word_masking
        self.tokenizer_path = self.args.tokenizer_path or self.args.model_name_or_path

    def setup(self, stage):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)

        if get_rank() == 0:
            logger.debug(f"Loaded tokenizer: {tokenizer}")

        tokenizer_name = self.tokenizer_path.rstrip("/").replace("/", "_")
        tokenize_fn = make_tokenize_function(tokenizer, self.args.max_sequence_length)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(tokenize_fn)

        tokenized_data_dir = str(self.data_dir / "tokenized")

        cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.dev_file}.seq_len_{self.args.max_sequence_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_{tokenize_fn_hash}.arrow",
        )
        maybe_cache_path = os.path.join(
            tokenized_data_dir,
            f"{self.args.train_file}.{self.args.dev_file}.seq_len_{self.args.max_sequence_length}.tokenizer_{tokenizer_name}.tokenize_fn_hash_.*.arrow",
        )
        maybe_cache_path_match_list = glob.glob(maybe_cache_path)
        logger.info(f"Rank {get_rank()} | Cache path: {cache_path}")

        with main_process_first(description="Loading dataset", active=(self.args.num_devices > 1)):
            if os.path.exists(cache_path):
                logger.success(f"Rank {get_rank()} | Found cached processed dataset: {cache_path}")
                processed_datasets = datasets.load_from_disk(cache_path)
                logger.success(
                    f"Rank {get_rank()} | Loaded cached processed dataset: {processed_datasets}"
                )
            elif len(maybe_cache_path_match_list) > 0 and os.path.exists(
                maybe_cache_path_match_list[0]
            ):
                logger.warning(
                    f"Rank {get_rank()} | Did not find cached processed dataset: {cache_path} but {maybe_cache_path_match_list[0]}. The tokenize function hash can change with small, functionally meaningless code changes in the tokenizers library. Proceeding with existing found cache."
                )
                processed_datasets = datasets.load_from_disk(maybe_cache_path_match_list[0])
                logger.success(
                    f"Rank {get_rank()} | Loaded cached processed dataset: {processed_datasets}"
                )
            else:
                processed_datasets = self.load_and_process_dataset(tokenizer, tokenized_data_dir)
                logger.info(f"Saving dataset to {cache_path}...")
                processed_datasets.save_to_disk(
                    cache_path, num_proc=self.args.preprocessing_workers
                )
        pad_to_multiple_of = 8 if self.args.precision in ["16-mixed", "bf16-mixed"] else None
        if self.args.language_modeling_strategy == "clm":
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of
            )
        elif self.args.language_modeling_strategy == "mlm":
            DataCollatorClass = (
                DataCollatorForWholeWordMask
                if self.whole_word_masking
                else DataCollatorForLanguageModeling
            )
            data_collator = DataCollatorClass(
                tokenizer=tokenizer,
                mlm=True,
                mlm_probability=self.mlm_probability,
                pad_to_multiple_of=pad_to_multiple_of,
            )

        self.train_dataset = processed_datasets["train"]
        self.dev_dataset = processed_datasets["dev"]
        self.data_collator = data_collator

        if self.args.data_preprocessing_only:
            exit(0)

    def load_and_process_dataset(self, tokenizer, tokenized_data_dir):
        extension = self.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {"train": self.train_file, "dev": self.dev_file}

        logger.info("Loading raw dataset...")
        tmp_load_dataset_cache_dir = (
            tempfile.mkdtemp(dir=tokenized_data_dir) if self.args.conserve_disk_space else None
        )
        train_dev_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            name=str(self.data_dir).replace("/", "_"),
            num_proc=self.args.preprocessing_workers,
            cache_dir=tmp_load_dataset_cache_dir,
        )

        if get_rank() == 0:
            logger.debug((train_dev_datasets, train_dev_datasets["train"][:2]))

        if self.args.conserve_disk_space:
            datasets.fingerprint.disable_caching()

        if self.args.line_by_line:
            processed_datasets = self.process_dataset_line_by_line(
                tokenizer=tokenizer,
                tokenizer_path=self.tokenizer_path,
                train_dev_datasets=train_dev_datasets,
            )
        else:
            processed_datasets = self.process_dataset_in_chunks(
                tokenizer=tokenizer, train_dev_datasets=train_dev_datasets
            )

        # processed_datasets["train"] = processed_datasets["train"].shuffle(seed=self.misc_args.seed) # <-- this is bad, triggers super expensive .flatten_indices op when .save_to_disk
        logger.success(
            f"Rank {get_rank()} | Finished processing datasets: {processed_datasets} | First sample len: {len(processed_datasets['train'][0]['input_ids'])}"
        )

        if self.args.conserve_disk_space:
            logger.info("Cleaning dataset loading cache...")
            try:
                shutil.rmtree(tmp_load_dataset_cache_dir)
            except OSError as e:
                # Reraise unless ENOENT: No such file or directory
                # (ok if directory has already been deleted)
                if e.errno != errno.ENOENT:
                    raise

            datasets.fingerprint.enable_caching()

        return processed_datasets

    def process_dataset_in_chunks(self, tokenizer, train_dev_datasets):
        """Expects input data to be one document per line. Tokenizes the documents and splits into chunks of max_sequence_legth."""
        tokenized_datasets = train_dev_datasets.map(
            make_tokenize_function(tokenizer, max_seq_length=None, truncate=False),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc="Running tokenizer on every text in dataset",
        )

        processed_datasets = tokenized_datasets.map(
            make_group_text_function(self.args.max_sequence_length),
            batched=True,
            batch_size=16_000,
            num_proc=self.args.preprocessing_workers,
            load_from_cache_file=not self.args.overwrite_data_cache,
            desc=f"Grouping texts in chunks of {self.args.max_sequence_length}",
        )

        return processed_datasets

    def process_dataset_line_by_line(self, tokenizer, tokenizer_path, train_dev_datasets):
        tokenized_data_dir = self.data_dir / "tokenized" / tokenizer_path
        os.makedirs(tokenized_data_dir, exist_ok=True)

        tokenize_fn = make_tokenize_function(tokenizer, self.args.max_sequence_length)
        tokenize_fn_hash = datasets.fingerprint.Hasher.hash(tokenize_fn)
        final_tokenized_filenames = {
            "train": os.path.join(
                tokenized_data_dir,
                f"seq_len_{self.args.max_sequence_length}.tokenize_fn_hash_{tokenize_fn_hash}.{self.args.train_file}",
            ),
            "dev": os.path.join(
                tokenized_data_dir,
                f"seq_len_{self.args.max_sequence_length}.tokenize_fn_hash_{tokenize_fn_hash}.{self.args.dev_file}",
            ),
        }
        cache_exists = os.path.exists(final_tokenized_filenames["train"]) and os.path.exists(
            final_tokenized_filenames["dev"]
        )
        logger.debug(
            f"Rank {get_rank()} | {tokenizer_path} | Cache exists: {cache_exists} | {'Loading cache...' if cache_exists else 'Starting dataset tokenization...'}"
        )

        # Always load from cache when not main process, dataset was already processed in main process
        load_from_cache = get_rank() != 0 or not self.args.overwrite_data_cache
        processed_datasets = train_dev_datasets.map(
            make_tokenize_function(tokenizer, self.args.max_sequence_length),
            batched=True,
            num_proc=1,  # Should use only one process to leverage tokenizers parallelism
            remove_columns=["text"],
            load_from_cache_file=load_from_cache,
            cache_file_names=final_tokenized_filenames,
            desc="Tokenizing dataset...",
        )

        return processed_datasets

    def train_dataloader(self):
        # TODO transfer common args to attribute and only do update shuffle for training
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=(True if self.args.workers > 0 else False),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc_args.too_many_open_files_fix
            else None,
            shuffle=True,
        )
        return DataLoader(self.train_dataset, collate_fn=self.data_collator, **common_args)

    def val_dataloader(self):
        common_args = dict(
            batch_size=self.args.batch_size_per_device,
            num_workers=self.args.workers,
            persistent_workers=(True if self.args.workers > 0 else False),  # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
            pin_memory=True,
            worker_init_fn=set_torch_file_sharing_strategy_to_system
            if self.misc_args.too_many_open_files_fix
            else None,
        )
        return DataLoader(self.dev_dataset, collate_fn=self.data_collator, **common_args)


def make_tokenize_function(tokenizer, max_seq_length=None, truncate=True):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=truncate,
            max_length=max_seq_length,
            # We use return_special_tokens_mask=True because DataCollatorForLanguageModeling is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    return tokenize_function


def make_group_text_function(max_seq_length):
    """Needs to be outside of DataModule because of hashing error in dataset.map"""

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    return group_texts
