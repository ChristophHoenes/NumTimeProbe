import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

import torch

from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.tapas_model import tapas_model_type_info, tapas_model, tapas_config, tapas_generation
from numerical_table_questions.tapex_model import tapex_model_type_info, tapex_model, tapex_config


@dataclass
class ModelTypeInfo:
    model_name_or_path: str
    pad_token_id: Optional[int] = None
    mask_token_id: Optional[int] = None
    # index of loss in outputs
    loss_out_id: Optional[Union[int, str]] = None
    prediction_scores_out_id: Optional[Union[int, str]] = 0
    hidden_state_out_id: Optional[Union[int, str]] = None
    attention_out_id: Optional[Union[int, str]] = None
    # whether the forward pass expects the tragets or not
    input_targets: bool = False
    # in data preparation if dataset is converted to TensorDataset filter
    # the specified attributes from the data dict
    filter_data_attributes: Optional[List[str]] = None
    # any additional model specific arguments
    # (if key is of type int it is interpreted as positional or otherwise as keyword argument)
    tensor_input_mapping: dict = field(default_factory=dict)  # TensorDataset serialization
    dict_input_mapping: dict = field(default_factory=dict)  # Dict-style (huggingface datasets) serialization

    def __post_init__(self):
        # TODO value checks
        pass


def get_model_type_info(model_name_or_path: str):
    match model_name_or_path.lower():
        case 'tapex':
            return ModelTypeInfo(**tapex_model_type_info())
        case 'tapas':
            return ModelTypeInfo(**tapas_model_type_info())
        case _:
            warnings.warn(f"Unknown model '{model_name_or_path}'! No ModelTypeInfo will be defined, relying on default config.")


def get_model_module(model_name: str):
    match model_name.lower():
        case 'tapex':
            model = tapex_model()
        case 'tapas':
            model = tapas_model()
        case _:
            # TODO try search path
            raise NotImplementedError(f"No initialization implemented for model {model_name}!")
    return model


def get_model_specific_config(model_name: str) -> dict:
    model_type_info = get_model_type_info(model_name)
    match model_name.lower():
        case 'tapex':
            other_kwargs = tapex_config()
        case 'tapas':
            other_kwargs = tapas_config()
        case _:
            other_kwargs = {}  # use default values of kwargs
    return {'model_type_info': model_type_info, **other_kwargs}


def map_batch_keys_to_model_kwarg(input_dict: dict, mapping: Dict[str, str] = {}, target=None) -> dict:
    #print("in key filter/mapping")
    if len(mapping) == 0:
        #print("provided mapping empty")
        # if no explicit mapping is provided by default pass all keys that were returned from the tokenizer
        if input_dict.get('tokenizer_keys') is not None:
            #print('selected tokenizer_keys:', input_dict['tokenizer_keys'])
            return {tokenizer_key: input_dict[tokenizer_key] for tokenizer_key in input_dict['tokenizer_keys']}
        return input_dict  # if no mapping and no explicit tokenizer_keys are provided return input_dict unchanged

    # route the correct data to the correct model input name (according to mapping from batch keys to model kwarg keys)
    #print("return filtered according to mapping: ", list(mapping.keys))
    return {model_input_name: input_dict[data_field_name]
            # 'targets' is a special keyword and will fetch the variable/argument target rather than a key from the original batch input
            if data_field_name != 'targets' else target
            for model_input_name, data_field_name in mapping.items()
            # only pass targets when explicitly defined; otherwise skip
            if not (data_field_name == 'targets' and target is None)
            # ignore/skip positional args (= int key)
            if isinstance(model_input_name, str)
            }


def get_sample_from_batch_dict(batch_dict: dict, idx: int, keep_dim: bool = True) -> dict:
    """ For every key in the dict style batch select the sample at position idx.
        If keep_dim=True fill restore the batch_size dimension (length 1).
    """
    return {key: batch_dict[key][idx].unsqueeze(dim=0)
            if keep_dim and isinstance(batch_dict[key], torch.Tensor)
            else batch_dict[key][idx]
            for key in batch_dict.keys()
            }


def model_specific_generation(model_name, model, tokenizer, inputs, outputs=None, **kwargs) -> List[str]:
    """
        outputs are the outputs of a regular forward pass (some models e.g. tapas need it for generation).
    """
    match model_name.lower():
        case 'tapas':
            # current implementation only works with dict style batch
            if not isinstance(inputs, dict):
                raise TypeError(f"Expected inputs to be dict type but found {type(inputs)}!")
            # filter batch input fields to only contain inputs required by TAPAS
            model_specific_inputs = map_batch_keys_to_model_kwarg(inputs, get_model_type_info('tapas').dict_input_mapping)
            # get table as dataframe from dataset table indices
            table_dataset = kwargs['test_dataset'].table_dataset  # assumes Dataset in test DataLoader to be of type QuestionTableIndexDataset
            table_data = [table_dataset.select([table_idx])[0]['table'] for table_idx in inputs['table_idx']]
            tables = [Table.from_state_dict(sample) for sample in table_data]
            return [tapas_generation(tokenizer,
                                     get_sample_from_batch_dict(model_specific_inputs, t),  # select only one sample for current table (for all keys)
                                     get_sample_from_batch_dict(outputs, t),  # select only one sample for current table (for all keys)
                                     table.pandas_dataframe
                                     )[0]  # as only one sample at a time is processed the returned list only contains one answer
                    for t, table in enumerate(tables)
                    ]
        case _:  # default generation via beam search (e.g. for tapex)
            answer_ids = model.generate(inputs['input_ids'] if isinstance(inputs, dict) else inputs[0],  # get input_ids from inputs depending on inputs type
                                        num_beams=kwargs.get('num_beams', 2),
                                        min_length=kwargs.get('min_length', 0),
                                        max_length=kwargs.get('max_length',
                                                              # if this is 1 it can lead to errors during generation -> set to at least 2
                                                              max(2, kwargs.get('max_target_len', 20))
                                                              ),
                                        no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 0),
                                        early_stopping=kwargs.get('early_stopping', False),
                                        )
            return tokenizer.batch_decode(answer_ids,
                                          skip_special_tokens=kwargs.get('skip_special_tokens', True),
                                          clean_up_tokenization_spaces=kwargs.get('clean_up_tokenization_spaces', True),
                                          )
