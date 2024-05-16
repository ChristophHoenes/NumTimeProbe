import copy
import errno
import glob
import os
import shutil
import tempfile
import warnings
from collections.abc import Iterable
from functools import partial
from dataclasses import asdict
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, Tuple, List, Dict, Callable

import datasets
import lightning as L
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.auto.tokenization_auto import AutoTokenizer

from numerical_table_questions.data_caching import caching
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset, table_collate
from numerical_table_questions.model import get_model_type_info
from numerical_table_questions.tokenizer_utils import get_tokenizer, prepare_for_tokenizer
from dlib.frameworks.pytorch import (
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


def path_from_components(data_dir, table_corpus, dataset_name, split, model_name=None) -> str:
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
        full_path = path_from_components(data_dir, table_corpus, dataset_name, split, model_name=model_name)
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


def batch_end_of_sequence(batch: torch.Tensor, pad_token_id: int, sequence_dimension: int = 1) -> torch.Tensor:
    """ Returns the indices where the pad token occurs for the first time. """
    is_padding = batch == pad_token_id
    any_padding = is_padding.sum(dim=sequence_dimension) >= 1
    first_padding = is_padding.int().argmax(dim=sequence_dimension)
    return torch.where(any_padding, first_padding, batch.shape[-1])


def cast_to_reduced_int(ints: torch.Tensor, num_values: Optional[int] = None):
    """
        Selects the smallest possible torch dtype for ints representing an id mapping of size num_value.
        If num_values is None the amount of values (e.g. vocab size) is estimated by the maximum of the
        values in the tensor plus one (for id zero).
    """
    # if num_values is None infer the coding size
    if num_values is None:
        num_values = ints.max() + 1
    if num_values <= 2:
        cast_to = torch.bool
    elif num_values <= 128:
        cast_to = torch.int8
    elif num_values <= 256 and ints.min() >= 0:
        cast_to = torch.uint8
    elif num_values <= 32768:
        cast_to = torch.int16
    elif num_values <= 2_147_483_648:
        cast_to = torch.int32
    else:
        cast_to = torch.int64
    return ints.to(cast_to)


def apply_sequence_transform(seqence_data:  Union[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]],
                             transform_fn: Callable[[torch.Tensor], torch.Tensor],
                             field_names: Optional[List[str]] = None,
                             **kwargs
                             ) -> Union[torch.Tensor, Dict[torch.Tensor]]:
    """ Applies a sequence transform to every tensor in seqence_data.
        Has side effects on seqence_data if it is a dictionary (changes contents at key=field_names in place).
    """
    if isinstance(seqence_data, dict):
        if field_names is None:
            raise ValueError("Must specify to which fields (dict keys) the transform should be applied! But field_names was None, expected list of strings.")
        for field in field_names:
            logger.info(f"Processing field '{field}':")
            seqence_data[field] = transform_fn(seqence_data[field], **kwargs)
        return seqence_data
    else:
        return transform_fn(seqence_data, **kwargs)


def unbind_table_batch(bound_table_batches: Union[torch.Tensor, List[torch.Tensor]],
                       pad_token_id: int,
                       end_of_sequence_idxs: Optional[List[int]] = None,
                       sequence_dimension: int = 1,
                       ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """ Unbind single questions and aswers from a table batch and remove batch padding. """
    logger.info("Unbinding table batches...")
    # wrap in list if single tensor (only one table batch)
    if isinstance(bound_table_batches, torch.Tensor):
        bound_table_batches = [bound_table_batches]
    # if the end of sequence ids are not provided calculate them based on the first occurance of padding token
    if end_of_sequence_idxs is None:
        end_of_sequence_idxs = [batch_end_of_sequence(batch, pad_token_id, sequence_dimension) for batch in bound_table_batches]
    # flatten rows (questions) in each (table) batch and discard everything after the respective entry in end_of_sequence_idxs (first pad token)
    batch_unbound = [row[:end_of_sequence_idxs[b][r]] for b, batch in enumerate(bound_table_batches) for r, row in enumerate(batch)]
    return batch_unbound, end_of_sequence_idxs


def truncate(tokenized: Union[torch.Tensor, List[torch.Tensor]],
             truncation_type: Union[bool, str],
             max_sequence_length: int,
             query_first: bool = True,
             num_reserved: Optional[Union[int, Iterable[Iterable[int]]]] = None,
             ):
    logger.info(f"Applying truncation strategy '{truncation_type}'...")
    # wrap in list if single tensor (only one table batch)
    if isinstance(tokenized, torch.Tensor):
        tokenized = [tokenized]
    # TODO option to drop tokens using heuristic such that question is still solvable (e.g drop unused column)
    if truncation_type in (True, 'longest_first'):
        truncated = [table_question[:max_sequence_length] if query_first else table_question[-max_sequence_length:]
                     for table_question in tokenized
                     ]
    elif truncation_type in (False, 'do_not_truncate'):
        # determine how many steps in each sequence to reserve for additional input
        if num_reserved is None:
            num_reserved = 0
        if isinstance(num_reserved, int):
            num_reserved = [[num_reserved] * batch.shape[-1] for batch in tokenized]
        # filter out sequences larger than model's max_length
        truncated = [
            table_question
            for b in range(len(tokenized))
            for i, table_question in enumerate(tokenized)
            if tokenized[b][i].shape[-1] + num_reserved[b][i] <= max_sequence_length
        ]
    else:
        # for compatibility with huggingface implement only_first & only_second but not needed up to now
        # TODO think about value error and check allowed argument values?
        raise NotImplementedError(f"Truncation strategy '{truncation_type}' is not implemented!")
    return truncated


def pad(tokenized: Union[torch.Tensor, List[torch.Tensor]],
        padding_type: Union[bool, str],
        max_sequence_length: int,
        pad_token_id: int,
        sequence_dimension: int = 1,
        ):
    logger.info(f"Applying padding strategy '{padding_type}'...")
    # wrap in list if single tensor (only one table batch)
    if isinstance(tokenized, torch.Tensor):
        tokenized = [tokenized]
    if padding_type in (True, 'max_length', 'longest'):
        if padding_type in (True, 'max_length'):
            # infer shape of tokenized tensors (or table batches) and change the sequence_dimension
            # to the maximum context length of the model
            tensor_shape = list(tokenized[0].shape)
            tensor_shape[sequence_dimension] = max_sequence_length
            # append dummy tensor to ensure maximum length sequence is present
            tokenized.append(torch.zeros(tensor_shape))
        # pad to longest sequence and unbind to list of single samples again
        padded = list(
            torch.unbind(
                torch.nn.utils.rnn.pad_sequence(
                    [table_question for table_question in tokenized],
                    batch_first=True,
                    padding_value=float(pad_token_id),
                    )
                )
            )
        # for True and 'max_length' pop dummy tensor that was previously appended
        if padding_type in (True, 'max_length'):
            padded = padded[:-1]
    elif padding_type in (False, 'do_not_pad'):
        padded = tokenized
    else:
        raise NotImplementedError(f"Padding strategy '{padding_type}' is not implemented!")
    return padded


def post_tokenizing(tokenized: dict, tokenizing_args: dict, max_sequence_length: int, pad_token_id: int, mask_token_id: int):
    truncated = truncate(tokenized,
                         tokenizing_args['truncation'],
                         max_sequence_length,
                         query_first=tokenizing_args['query_first'],
                         allow_custom_truncation=tokenizing_args['allow_custom_truncation'],
                         )

    # save input lengths to know at what index to inject target masks for every sample
    input_lengths = [table_question.shape[-1] for table_question in truncated['input_ids']]
    target_lengths = [table_question.shape[-1] for table_question in truncated['targets']]

    padded = pad(truncated,
                 tokenizing_args['padding'],
                 max_sequence_length,
                 pad_token_id,
                 )
    # fill mask tokens for target predictions
    target_dummy = torch.ones_like(padded['input_ids'][0]) * mask_token_id
    full_sequence_targets = []
    for i, (input_end_idx, target_length) in enumerate(zip(input_lengths, target_lengths)):
        # inputs do not need a label mask
        #padded['input_ids'][i][input_end_idx:input_end_idx+target_length] = mask_token_id
        template = torch.clone(target_dummy)
        template[:target_length] = padded['targets'][i][:target_length]
        full_sequence_targets.append(template)
    padded['targets'] = full_sequence_targets
    return padded


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

    def prepare_data(self):
        # download not needed as locally on disk from data_synthesis
        if self.lazy_data_processing:
            return  # no preparation needed if it is doene on the fly during data loading
        # but once published download from huggingface datasets
        for split in ['train', 'validation', 'test']:
            # path definitions to check for saved files
            huggingface_base_dir = f"{self.table_corpus}_{self.dataset_name}_{self.model_name}_tokenized"
            final_processing_path = Path(self.data_dir) / 'viable_tensors' / huggingface_base_dir / split
            intermediate_processing_path = Path(self.data_dir) / 'full_dict' / huggingface_base_dir / split
            if final_processing_path.exists() and not self.overwrite_cache:
                # load fully processed tensor dataset to ensure no error occurs
                data_split = datasets.load_from_disk(final_processing_path)
            elif intermediate_processing_path.exists() and not self.overwrite_cache:
                # load from intermediate step (all examples) and apply custom post-processing and filtering
                tokenized_dict = datasets.load_from_disk(intermediate_processing_path).with_format('torch')
                # convert to dict while keeping the tensor format
                tokenized_dict = {field: [tokenized_dict[i][field] for i in range(tokenized_dict.num_rows)]
                                  for field in tokenized_dict.column_names
                                  }
                processed_sequences = post_tokenizing(tokenized_dict,
                                                      self.tokenizing_args,
                                                      self.max_num_tokens,
                                                      self.model_specs.pad_token_id,
                                                      self.model_specs.mask_token_id,
                                                      )
                # save fully processed dataset
                datasets.Dataset.from_dict(processed_sequences).save_to_disk(
                    self.data_dir
                    + '/viable_tensors/'
                    + huggingface_base_dir
                    + f"/{split}"
                )
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
                # transform input to format expected by tokenizer (only considering input and target fields)
                tokenizer_inputs = prepare_for_tokenizer(data_split, self.model_name, **tokenizing_args)
                logger.info("Tokenize examples...")
                tokenized = [(self.tokenizer(**table_question), self.tokenizer(**target)['input_ids'])
                             for table_question, target in tqdm(tokenizer_inputs)]
                tokenizer_output_names = tokenized[0][0].keys()
                # convert tuple of tokenized model inputs (outputs depend on tokenizer)
                # and target token_ids to dict of tensors and infer smallest possible int dtype to represent the ids
                tokenized_dict = {key: [cast_to_reduced_int(sample[0][key], num_values=self.tokenizer.vocab_size)
                                        # for tokenized (non-mask) outputs pick smallest possible int dtype for the vocab_size
                                        if 'mask' not in key
                                        # for masks infer the coding size (most likely binary)
                                        else cast_to_reduced_int(sample[0][key])
                                        for sample in tokenized
                                        ]
                                  for key in tokenizer_output_names}
                tokenized_dict['targets'] = [cast_to_reduced_int(sample[1], self.tokenizer.vocab_size) for sample in tokenized]

                # add other fields of data split that did not go through the tokenizer
                missing_fields = list(set(data_split.column_names)
                                      - set(['table'])  # do not copy table for each question
                                      - set(tokenized_dict.keys())
                                      )
                additional_fields_dict = {field: data_split[field] for field in missing_fields}
                # since table was removed from keys for efficiency, add column with table_id for reference (repeat id for each question to the table)
                additional_fields_dict['table_id'] = [[table_batch['table']['table_id']] * len(table_batch['questions'])
                                                      for table_batch in data_split
                                                      ]
                # TODO table num rows, table num cols, unique values aggregation column
                tokenized_dict.update(additional_fields_dict)
                # flatten the table batches to sequence of questions
                tokenized_dict = unbind_table_batch(tokenized_dict, self.model_specs.pad_token_id)

                # save raw tokenizer outputs (sequences with variable length)
                datasets.Dataset.from_dict(tokenized_dict).save_to_disk(
                    self.data_dir
                    + '/full_dict/'
                    + huggingface_base_dir
                    + f"/{split}"
                )
                # save as metadata (in extra text file) the length of the dataset before post_tokenizing (e.g. before filtering too long sequences)
                num_samples_before_filtering = len(tokenized_dict['input_ids'])
                with (Path(self.data_dir) / 'full_dict' / huggingface_base_dir / split / 'custom_metadata.txt').open('a+') as f:
                    f.write(f"{datetime.now().strftime('%y%m%d_%H%M_%S_%f')} num_rows {num_samples_before_filtering}\n")

                processed_sequences = post_tokenizing(tokenized_dict,
                                                      self.tokenizing_args,
                                                      self.max_num_tokens,
                                                      self.model_specs.pad_token_id,
                                                      self.model_specs.mask_token_id,
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
                self.splits['train'] = QuestionTableIndexDataset(path_from_components(self.data_dir, self.table_corpus, self.dataset_name, 'train'))
                self.splits['validation'] = QuestionTableIndexDataset(path_from_components(self.data_dir, self.table_corpus, self.dataset_name, 'validation'))
            else:
                self.splits['train'] = load_split_tensor('train', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('train')
                self.splits['validation'] = load_split_tensor('validation', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('validation')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            if self.lazy_data_processing:
                self.splits['test'] = QuestionTableIndexDataset(path_from_components(self.data_dir, self.table_corpus, self.dataset_name, self.model_name, 'test'))
            else:
                self.splits['test'] = load_split_tensor('test', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('test')

        if stage == 'predict':
            if self.lazy_data_processing:
                self.splits['test'] = QuestionTableIndexDataset(path_from_components(self.data_dir, self.table_corpus, self.dataset_name, self.model_name, 'test'))
            else:
                self.splits['test'] = load_split_tensor('test', self.table_corpus, self.dataset_name, self.model_name, self.data_dir, output_dict=self.is_batch_dict)
                check_dataset_type('test')

    def _get_dataloader(self, split_name: str, split_config: dict) -> DataLoader:
        # determine collate function for processing during data loading
        if self.lazy_data_processing:
            collate_fn = partial(
                table_collate,
                model_name=self.model_name,
                tokenizer=self.tokenizer,
                truncation=self.tokenizing_args['truncation'],
                padding=self.tokenizing_args['padding']
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
