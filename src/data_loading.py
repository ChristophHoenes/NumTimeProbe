import errno
import glob
import os
import shutil
import tempfile
import warnings
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import datasets
import lightning as L
import torch
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, TapexTokenizer
from transformers.data.data_collator import DataCollatorForWholeWordMask
from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.data_caching import caching
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


def prepare_for_tokenizer(data, model_name, **kwargs):
    # TODO implement model specific pre-tokenizer formatting and default case
    # TODO maybe a dict of functions instead of long if/elif/else would be cleaner
    if model_name == 'tapex':
        # TODO try performance with list of tables (duplicated) references as well
        # TODO flatten each question to sample also considering alternative answers
        logger.info("Flattening examples to tokenizer format...")
        padding = kwargs.get('padding') or True
        truncation = kwargs.get('truncation') or True
        max_length = kwargs.get('max_length') or 512
        return_tensors = kwargs.get('return_tensors') or 'pt'
        # TODO try batch encode for significant speedup (multiple questions per table)
        questions_by_table = {}
        for question in data._questions:
            if questions_by_table.get(question._table._table_id) is None:
                questions_by_table[question._table._table_id] = {'questions': [question._nl_question],
                                                                 # TODO handle string conversion elsewhere
                                                                 'answers': [str(question._answer)]}
            else:
                questions_by_table[question._table._table_id]['questions'].append(question._nl_question)
                # TODO handle string conversion elsewhere
                questions_by_table[question._table._table_id]['answers'].append(str(question._answer))
        return [
            (
                {
                    'table': data._tables[table_id].pandas_dataframe,
                    'query': content_dict['questions'],
                    'padding': padding,
                    'truncation': truncation,
                    'max_length': max_length,
                    'return_tensors': return_tensors,
                },
                {
                    'answer': content_dict['answers'],
                    'padding': padding,
                    'truncation': truncation,
                    'max_length': max_length,
                    'return_tensors': return_tensors,
                    'add_special_tokens': False,
                }
            )
            for table_id, content_dict in tqdm(questions_by_table.items())
        ]
    else:
        raise NotImplementedError(f"No tokenization behavior implemented for model '{model_name}'!")


def get_tokenizer(model_name, **kwargs):
    if model_name == 'tapex':
        return TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
    elif model_name == 'omnitab':
        return AutoTokenizer.from_pretrained("neulab/omnitab-large-finetuned-wtq")
    else:
        raise NotImplementedError(f"No tokenizer getter implemented for model '{model_name}'!")


# TODO consider moving within TableQADataModule -> less arguments as most contained in self but reuse outside of class needed?
def load_split_tensor(split: str, dataset_name: str, model_name: str, data_dir: str = './data/NumTabQA/.cache'):
    # data_dict = pickle.load((Path(data_dir) / f"viable_tensors_{split}_{dataset_name}_{model_name}_tokenized.pickle").open('rb'))
    path = Path(data_dir) / 'viable_tensors' / f"{dataset_name}_{model_name}_tokenized" / split
    data_dict = datasets.Dataset.load_from_disk(path)
    # targets_tensor = data_dict.get('targets')
    try:
        targets_tensor = torch.tensor(data_dict['targets'])
    except KeyError:
        targets_tensor = None
    # inputs_tensors = [value for key, value in data_dict.items() if key != 'targets']
    inputs_tensors = [torch.tensor(data_dict[col]) for col in data_dict.column_names if col != 'targets']
    inputs_dataset = torch.utils.data.TensorDataset(*inputs_tensors)
    if targets_tensor is not None:
        return torch.utils.data.StackDataset(inputs_dataset, torch.utils.data.TensorDataset(targets_tensor))
    return inputs_dataset


def batch_end_of_sequence(batch, pad_token_id):
    """ Returns the indices where the pad token occurs for the first time. """
    is_padding = batch == pad_token_id
    any_padding = is_padding.sum(dim=1) >= 1
    first_padding = is_padding.int().argmax(dim=1)  # TODO is -1 more general?
    return torch.where(any_padding, first_padding, batch.shape[-1])


def cast_to_reduced_int(ints, num_values: int = None, field_name: Optional[str] = None):
    """
        Selects the smallest possible torch dtype for ints representing an id mapping of size num_value.
        If num_values is None the amount of values (e.g. vocab size) is estimated by the maximum of the
        values in the tensor plus one (for id zero).
    """
    if 'mask' in field_name:
        num_values = None
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


def unbind_table_batch(bound_table_batches, pad_token_id):
    """ Unbind single questions and aswers from a table batch and remove batch padding. """
    # TODO maybe remove also end of sequence token in input before target mask? - I think not
    logger.info("Unbinding table batches...")
    # save the end of sequence of input_ids (last id before first padding) for every batch
    end_of_sequence_idxs = []
    # also save the original sequence length of every batch (before removing padding)
    input_seq_len = []
    # manually ensure the order such that all fields are processed like input_ids except target_ids
    # (which then changes the state of end_of_sequence_idxs & input_seq_len)
    tokenizer_keys = list(bound_table_batches.keys())
    tokenizer_keys.remove('input_ids')
    tokenizer_keys.remove('targets')
    tokenizer_keys = ['input_ids', *tokenizer_keys, 'targets']
    for key in tokenizer_keys:
        batches_unbound = []
        value = bound_table_batches[key]
        for b, batch in enumerate(value):
            if key in ['input_ids', 'targets']:
                # save the states of input_ids to apply them to all other fields
                # only recompute for targets which have different sequence lengths
                end_of_sequence_idx = batch_end_of_sequence(batch, pad_token_id)
                end_of_sequence_idxs.append(end_of_sequence_idx)
                input_seq_len.append(bound_table_batches['input_ids'][b].shape[-1])
            else:
                # lookup input_ids state
                end_of_sequence_idx = end_of_sequence_idxs[b]
            # test if the field is a tensor with the same sequence length as the input tokens
            is_like_input_sequence = (
                isinstance(bound_table_batches[key][0], torch.Tensor)
                and bound_table_batches[key][b].shape[-1] == input_seq_len[b]
            )  # define condition before if clause for readability
            if is_like_input_sequence or key == 'targets':
                # only apply padding removal for fields that relate to input_ids
                batch_unbound = [row[:end_of_sequence_idx[i]] for i, row in enumerate(batch)]
            else:
                # leave unchanged
                batch_unbound = batch
            batches_unbound.append(batch_unbound)
        bound_table_batches[key] = [question for table_batch in batches_unbound for question in table_batch]
    return bound_table_batches


def truncate(tokenized: dict[torch.Tensor],
             truncation_type: Union[bool, str],
             max_sequence_length: int,
             query_first: bool = True,
             allow_custom_truncation: bool = True,
             ):
    logger.info("Apply truncation strategy...")
    # TODO option to drop tokens using heuristic such that question is still solvable (e.g drop unused column)
    if truncation_type in (True, 'longest_first'):
        truncated = {}
        for tokenizer_output in tokenized.keys():
            is_like_input_sequence = (
                isinstance(tokenized[tokenizer_output][0], torch.Tensor)
                and tokenized[tokenizer_output][0].shape[-1] == tokenized['input_ids'][0].shape[-1]
            )
            if is_like_input_sequence:
                truncated[tokenizer_output] = [
                    torch.narrow(
                        table_question,
                        -1,  # last dimension = sequence length
                        # if query_first truncate at the end of input and vice versa
                        0 if query_first else -1,
                        # input and target jointly need to fit in max_num_tokens
                        max_sequence_length - tokenized['targets'][i].shape[-1]
                        )
                    for i, table_question in enumerate(tokenized[tokenizer_output])
                    ]
            else:
                truncated[tokenizer_output] = tokenized[tokenizer_output]
    elif truncation_type in (False, 'do_not_truncate'):
        # filter out sequences larger than model's max_length
        truncated = {}
        for tokenizer_output in tokenized.keys():
            truncated[tokenizer_output] = [
                table_question
                for i, table_question in enumerate(tokenized[tokenizer_output])
                if tokenized['input_ids'][i].shape[-1] + tokenized['targets'][i].shape[-1] <= max_sequence_length
            ]
    elif allow_custom_truncation is True:
        truncated = tokenized
    else:
        # for compatibility with huggingface implement only_first & only_second but not needed up to now
        # TODO think about value error and check allowed argument values?
        raise NotImplementedError(f"Truncation strategy '{truncation_type}' is not implemented!")
    return truncated


def pad(tokenized:  dict[torch.Tensor],
        padding_type: Union[bool, str],
        max_sequence_length: int,
        pad_token_id: int,
        ):
    logger.info("Apply padding strategy...")
    if padding_type in (True, 'max_length'):
        padded = {}
        for tokenizer_output in tokenized.keys():
            is_like_input_sequence = (
                isinstance(tokenized[tokenizer_output][0], torch.Tensor)
                and tokenized[tokenizer_output][0].shape[-1] == tokenized['input_ids'][0].shape[-1]
            )
            if is_like_input_sequence and tokenizer_output:
                if tokenizer_output != 'targets':
                    # determine shape of tensor and alter last dimension to maximum sequence length of model
                    tensor_shape = list(tokenized[tokenizer_output][0].shape)
                    tensor_shape[-1] = max_sequence_length
                    # append dummy tensor to ensure maximum length sequence is present
                    tokenized[tokenizer_output].append(torch.zeros(tensor_shape))
                # pad to longest sequence and unbind to list of single samples again
                padded[tokenizer_output] = list(torch.unbind(
                    torch.nn.utils.rnn.pad_sequence(
                        [table_question for table_question in tokenized[tokenizer_output]],
                        batch_first=True,
                        # TODO check semantics of pad token for masks
                        padding_value=float(pad_token_id) if 'mask' not in tokenizer_output else 0.,
                    )
                ))[:-1]  # pop dummy tensor that was previously appended
            else:
                # TODO think of removing unbind and instead move vstack here such that
                # by convention after padding there is a tensor instead of a list of samples
                padded[tokenizer_output] = tokenized[tokenizer_output]
    elif padding_type in (False, 'do_not_pad'):
        padded = tokenized
    else:
        raise NotImplementedError(f"Padding strategy '{padding_type}' is not implemented!")


def post_tokenizing(tokenized: dict[torch.Tensor], tokenizing_args: dict, max_sequence_length: int, pad_token_id: int, mask_token_id: int):
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
        padded['input_ids'][i][input_end_idx:input_end_idx+target_length] = mask_token_id
        template = torch.clone(target_dummy)
        template[input_end_idx:input_end_idx+target_length] = padded['targets'][i][:target_length]
        full_sequence_targets.append(template)
    padded['targets'] = full_sequence_targets
    return padded


class TableQADataModule(L.LightningDataModule):
    def __init__(self,
                 model,
                 tokenizing_args=None,
                 data_dir: str = './data/NumTabQA/.cache',
                 dataset_name='basic_dataset',
                 overwrite_cache=False,
                 ):
        super().__init__()
        self.model_specs = model.model_specs
        # TODO test if model name is known else raise NotImplemented error
        self.model_name = model.model_specs.model_name_or_path
        self.tokenizing_args = asdict(tokenizing_args) or dict()
        self.tokenizer = get_tokenizer(self.model_name, **self.tokenizing_args)
        self.max_num_tokens = self.tokenizing_args.get('max_length') or 512
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.overwrite_cache = overwrite_cache
        self.splits = dict()

    def prepare_data(self):
        # download not needed as locally on disk from data_synthesis
        # but once published download from huggingface datasets
        for split in ['test']:  # ['train', 'validation', 'test']: <- skip other splits while developing
            # path definitions to check for saved files
            huggingface_base_dir = f"{self.dataset_name}_{self.model_name}_tokenized"
            final_processing_path = Path(self.data_dir) / 'viable_tensors' / huggingface_base_dir / split
            intermediate_processing_path = Path(self.data_dir) / 'full_dict' / huggingface_base_dir / split
            if final_processing_path.exists() and not self.overwrite_cache and False:  # TODO remove and False kill switch
                # load fully processed tensor dataset
                data_split = datasets.load_from_disk(final_processing_path)
            elif intermediate_processing_path.exists() and not self.overwrite_cache and False:  # TODO remove and False kill switch
                # load from intermediate step (all examples) and apply custom post-processing and filtering
                tokenized_dict = datasets.load_from_disk(intermediate_processing_path)
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
                base_filename = f"{split}_{self.dataset_name}.pickle"
                data_split = caching(self.data_dir, base_filename)
                if data_split is None:
                    raise ValueError(f"No data split '{split}' found at {self.data_dir}! "
                                     "Please download, or generate the requested dataset.")
                # always disable padding and truncation - apply configuration afterwards
                # except for special truncation strategies supported by the tokenizer
                tokenizing_args = self.tokenizing_args.copy()
                tokenizing_args.update({'padding': False,
                                        'truncation': self.tokenizing_args['truncation']
                                        if self.tokenizing_args['allow_custom_truncation'] else False
                                        }
                                       )
                # transform input to format expected by tokenizer
                tokenizer_inputs = prepare_for_tokenizer(data_split, self.model_name, **tokenizing_args)
                logger.info("Tokenize examples...")
                tokenized = [(self.tokenizer(**table_question), self.tokenizer(**target)['input_ids'])
                             for table_question, target in tqdm(tokenizer_inputs)]
                # TODO remove following line after testing
                tokenizer_outputs = tokenized[0][0].keys()
                # convert tuple of tokenized model inputs (outputs depend on tokenizer)
                # and target token_ids to dict of tensors and infer smallest possible int dtype to represent the ids
                tokenized_dict = {key: [cast_to_reduced_int(sample[0][key], self.model_specs.vocab_size, field_name=key)
                                        for sample in tokenized]
                                  for key in tokenizer_outputs}
                tokenized_dict['targets'] = [cast_to_reduced_int(sample[1], self.model_specs.vocab_size) for sample in tokenized]
                tokenized_dict = unbind_table_batch(tokenized_dict, self.model_specs.pad_token_id)
                # save raw tokenizer outputs (sequences with variable length)
                datasets.Dataset.from_dict(tokenized_dict).save_to_disk(
                    self.data_dir
                    + '/full_dict/'
                    + huggingface_base_dir
                    + f"/{split}"
                )
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

    def setup(self, stage: str):
        print('setup', stage)

        def check_dataset_type(split_name):
            if not isinstance(self.splits[split_name], torch.utils.data.StackDataset):
                # TODO think of using TypeError instead
                warnings.warn(
                    f"Dataset should have type 'torch.utils.data.StackDataset' if there are targets but is of type '{type(self.splits[split_name])}'! "
                    "There should always be targets available check your dataset."
                )
                # raise TypeError(f"Dataset should have type 'torch.utils.data.StackDataset' but is of type '{type(self.splits[split_name])}'! "
                #                "Dataset should return a variable length tuple of model inputs and the targets as torch.Tensor.")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # TODO change path arguments to 'train' and 'validation' once computed
            self.splits['train'] = load_split_tensor('test', self.dataset_name, self.model_name, self.data_dir)
            check_dataset_type('train')
            self.splits['validation'] = load_split_tensor('test', self.dataset_name, self.model_name, self.data_dir)
            check_dataset_type('validation')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test':
            self.splits['test'] = load_split_tensor('test', self.dataset_name, self.model_name, self.data_dir)
            check_dataset_type('test')

        if stage == 'predict':
            self.splits['test'] = load_split_tensor('test', self.dataset_name, self.model_name, self.data_dir)
            check_dataset_type('test')

    def train_dataloader(self):
        if isinstance(self.splits['train'], torch.utils.data.TensorDataset):
            return WrapCustomTupleDataLoader(self.splits['train'], batch_size=64, custom_tuple=(None,))
        return DataLoader(self.splits['train'], batch_size=64)

    def val_dataloader(self):
        if isinstance(self.splits['train'], torch.utils.data.TensorDataset):
            return WrapCustomTupleDataLoader(self.splits['validation'], batch_size=64, custom_tuple=(None,))
        return DataLoader(self.splits['validation'], batch_size=64)

    def test_dataloader(self):
        if isinstance(self.splits['train'], torch.utils.data.TensorDataset):
            return WrapCustomTupleDataLoader(self.splits['test'], batch_size=64, custom_tuple=(None,))
        return DataLoader(self.splits['test'], batch_size=64)

    def predict_dataloader(self):
        if isinstance(self.splits['train'], torch.utils.data.TensorDataset):
            return WrapCustomTupleDataLoader(self.splits['test'], batch_size=64, custom_tuple=(None,))
        return DataLoader(self.splits['test'], batch_size=64)


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
