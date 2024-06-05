import warnings
from dataclasses import dataclass, field
from typing import Optional, Union, List

from numerical_table_questions.data_synthesis import Table
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
    input_mapping: dict = field(default_factory=dict)
    dict_input_mapping: dict = field(default_factory=dict)

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


def model_specific_generation(model_name, model, tokenizer, inputs, **kwargs):
    match model_name.lower():
        case 'tapas':
            # get required input arguments for tapas generation:
            # extract model input_ids from inputs depending on the batch type (although currently must be dict)
            input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs[0]
            # get model outputs
            model_outputs = model(inputs)
            # get table as dataframe from dataset table indices
            table_dataset = kwargs['test_dataset'].table_dataset  # assumes Dataset in test DataLoader to be of type QuestionTableIndexDataset
            table_data = [table_dataset.select([table_idx])['table'] for table_idx in inputs['table_idx']]
            table = Table.from_state_dict(table_data)
            table_df = table.pandas_dataframe
            return tapas_generation(tokenizer, input_ids, model_outputs, table_df)
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
