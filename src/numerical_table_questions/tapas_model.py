import math
import logging
import logging.config
import warnings
from pathlib import PurePath
from typing import List, Union, Tuple

import pandas as pd
import torch
import transformers
from tqdm import tqdm

from numerical_table_questions.answer_coordinates import AnswerCoordinates, compute_answer_coordinates
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet
from numerical_table_questions.metrics import str_match_accuracy


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def tapas_model_type_info() -> dict:
    return dict(
        model_name_or_path='tapas',
        pad_token_id=0,
        # TODO change variable name because in TAPEX and current code it does not mean the [MASK] token but rather a special id for attention mask in targets
        mask_token_id=None,
        input_targets=False,
        loss_out_id='loss',
        filter_data_attributes=None,  # only needed for TensorDataset serialization
        tensor_input_mapping={},  # only needed for TensorDataset serialization and positional arguments
        dict_input_mapping={},  # by default use all tokenizer keys as input
        )


def tapas_model(hf_version_path: str = "google/tapas-base-finetuned-wtq"):
    model = transformers.TapasForQuestionAnswering.from_pretrained(hf_version_path)
    # change model config
    model.config.return_dict = True
    return model


def tapas_config() -> dict:
    return dict(
        generation_metrics={
            'str_match_accuracy': (str_match_accuracy,
                                   {
                                    'post_processing_fn': lambda x: [item.strip() for item in x],
                                   }
                                   )
            }
        )


def tapas_tokenizer_format(data, lazy: bool = False, is_eval: bool = False, **kwargs):
    if isinstance(data, TableQuestionDataSet):
        raise NotImplementedError("Preparation of TAPAS tokenizer format is only implemented for huggingface datasets serialization!")
    else:
        if lazy:
            processed_samples = []
            for sample in data:
                #print(list(sample.keys()))
                table = Table.from_state_dict(sample['table'])
                table_df = table.pandas_dataframe
                # token_type_ids are created per column and the Embedding matrix of the TAPAs model currently has a fixed size which must not be exceeded
                # retrieve question
                question = sample['questions'][kwargs['question_number']]
                if not is_eval:  # for training Answer coordinates are required
                    answer_text = [sample['answers'][kwargs['question_number']]]  # should only be provided for evaluation
                    # if answer coordinates are not pre-computed do it here
                    if sample.get('answer_coordinates') is None:
                        # TODO compate to TAPAS utils scripts/notebook (https://github.com/NielsRogge/tapas_utils?tab=readme-ov-file)
                        # see also table normalization https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py
                        answer_coordinates = compute_answer_coordinates(
                            column_name=sample['aggregation_columns'][kwargs['question_number']],
                            dataframe=table_df,
                            sql_query=sample['sql'][kwargs['question_number']],
                            ).generate()
                    else:
                        answer_coordinates = AnswerCoordinates(**sample['answer_coordinates'][kwargs['question_number']]).generate()
                    answer_coordinates = [answer_coordinates]  # wrap in list analogous to question
                else:
                    answer_text = None
                    answer_coordinates = None
                processed_samples.append({
                    'table': table_df,
                    'queries': [question],
                    'answer_coordinates': answer_coordinates,
                    'answer_text': answer_text,
                    'padding': kwargs.get('padding', 'max_length'),
                    'truncation': kwargs.get('truncation', 'drop_rows_to_fit'),
                    'return_tensors': kwargs.get('return_tensors', 'pt'),
                })
            return processed_samples
        else:
            raise NotImplementedError("No Tapas tokenizer preparation implemented for non-lazy option!")


# from examples on https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/tapas#usage-inference
def get_aggregator_string(predicted_aggregation_indices) -> List[str]:
    id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    return [id2aggregation[x] for x in predicted_aggregation_indices]


# from examples on https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/tapas#usage-inference
def get_answer_cell_values(table: pd.DataFrame, predicted_answer_coordinates) -> List[str]:
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answers.append(table.iat[coordinates[0]])
        else:
            # multiple cells
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            answers.append(", ".join(cell_values))
    return answers


def string_to_num(number_string: str) -> Union[int, float]:
    try:
        return int(number_string)
    except ValueError:
        try:
            return float(number_string)
        except ValueError:
            return float('nan')


def compute_aggregation(aggregators, answer_cells) -> List[str]:
    results = []
    for agg, cells in zip(aggregators, answer_cells):
        parsed_cells = [string_to_num(cell) for cell in cells.split(', ')]
        match agg:
            case 'NONE':
                if len(parsed_cells) > 1:
                    warnings.warn("Encountered multi-row answer with aggregator NONE (NOOP)! Make sure the answer is a single value. "
                                  "Only the first value will be returned."
                                  )
                results.append(parsed_cells[0])
            case 'SUM':
                results.append(sum(parsed_cells))
            case 'AVERAGE':
                results.append(sum(parsed_cells)/len(parsed_cells))
            case 'COUNT':
                results.append(len(parsed_cells))
            case _:
                raise ValueError(f"Encountered unknown aggregator {agg}! Make sure the mapping defined in get_aggregator_string is correct.")
    # answers should be strings -> convert to correct number format as string
    postprocessed_number_format = ['nan'
                                   if math.isnan(number)  # special case for invalid numbers
                                   else str(number)
                                   for number in results
                                   ]
    return postprocessed_number_format


def tapas_generation(tokenizer, model_inputs: dict, model_outputs: dict, table: pd.DataFrame) -> List[str]:
    answer_coordinates, aggregation_indices = (
        tokenizer.convert_logits_to_predictions({key: value.detach().cpu()
                                                 for key, value in model_inputs.items()
                                                 if isinstance(value, torch.Tensor)
                                                 },  # assumes dict input
                                                model_outputs['logits'].detach().cpu(),  # assumes dict type for model_outputs not BatchEncoding
                                                model_outputs['logits_aggregation'].detach().cpu()  # assumes dict type for model_outputs not BatchEncoding
                                                )
        )
    aggregators = get_aggregator_string(aggregation_indices)
    answer_cells = get_answer_cell_values(table, answer_coordinates)

    predictions = compute_aggregation(aggregators, answer_cells)
    return predictions


"""
def reduce_answer_coordinates(tok_input: dict, last_cutoff: Optional[int] = None, is_success: bool = False, iteration: int = 0, max_length=512) -> int:
    if iteration > max_length + 1:
        raise StopIteration("Iteration exceeded max_length of model. Aborting due to risk of infinity loop.")

    def _max_row_id(answer_coordinates: List[Tuple[int, int]]) -> int:
        return max([coordinate[0] for coordinate in answer_coordinates])

    max_row_id = _max_row_id(tok_input['answer_coordinates'][0])

    if iteration == 0 or last_cutoff is None:

        # more efficient heuristic by evaluating always half length and adjust behavior based on output
        ## in first iteration remove all cells that exceed the max_length of the model
        #reduced_coordinates = [coordinate for coordinate in tok_input['answer_coordinates'][0]
        #                       if coordinate[0] * coordinate[1] <= max_length
        #                       ]
        ## if the previous step did reduce the answer coordinates return result else proceed with removing last row
        #if max_row_id != _max_row_id(reduced_coordinates):
        #    tok_input['answer_coordinates'][0] = reduced_coordinates

        # in first iteration test if dropping all rows up to the last answer row would fit into the model
        tok_input['answer_coordinates_reduced'] = [coordinate for coordinate in tok_input['answer_coordinates'][0]
                                                   if coordinate[0] <= max_row_id
                                                   ]
        return max_row_id
    else:
        if is_success
            new_cutoff = (last_cutoff + tok_input['table'].shape[0]) // 2  # half-way between max_success and min_fail
        # remove the row with highest index (max_row_id)
        tok_input['answer_coordinates'][0] = [coordinate for coordinate in tok_input['answer_coordinates'][0]
                                              if coordinate[0] < max_row_id
                                              ]
"""


def reduce_answer_coordinates(tokenizer, tokenizer_inputs, max_iteration=10):
    def _max_row_id(answer_coordinates: List[Tuple[int, int]]) -> int:
        return max([coordinate[0] for coordinate in answer_coordinates])

    def _compute_next_answer_coordinates(answer_coordinates: List[Tuple[int, int]],
                                         lower_bound: int,
                                         upper_bound: int,
                                         ) -> Tuple[List[Tuple[int, int]], int]:
        new_row_cutoff = (lower_bound + upper_bound) // 2
        new_answer_cells = [coordinate for coordinate in answer_coordinates
                            if coordinate[0] <= new_row_cutoff
                            ]
        return new_answer_cells, new_row_cutoff

    # for every sample narrow down the window of when tokenization fails/succeeds by trying to half the window size in every step
    # until the window collapses to a single point (the maximum number of answer cells before tokenization fails)
    for tok_input in tokenizer_inputs:
        if tok_input.get('answer_coordinates') is not None and tok_input['answer_coordinates'][0] is not None:
            iteration = 0
            min_fail = None  # the lowest row_id for which the tokenization failed (window lower bound)
            max_success = 0  # the highest row_id for which tokenizing completed sucessfully (window upper bound)
            new_row_cutoff = None  # last row_id that is included in the answer coordinate version that is tried next
            new_answer_cells = None  # reduced answer coordinates to try next
            while True:
                # prevent infinity loop
                if iteration > max_iteration + 1:
                    raise StopIteration("Iteration exceeded max_length of model. Aborting due to risk of infinity loop.")
                # try if tokenizing the example is sucessful
                try:
                    if new_answer_cells is None:
                        tokenizer(**tok_input)
                        break  # if tokenization successful leave loop
                    else:
                        # try with modified answer coordinates
                        tokenizer(**{key: value if key != 'answer_coordinates' else [new_answer_cells]
                                     for key, value in tok_input.items()
                                     }
                                  )
                        max_success = new_row_cutoff  # if success increase max_success to the cutoff that was successfully tried just now (shift window lower bound to right)
                except ValueError as e:
                    logger.info(f"Original Exception: {e}\nProbably table reduced to max answer coordinate row_id does not fit\n-> reducing answer coordinates iteration {iteration}...")
                    # code goes here
                    if iteration == 0:
                        min_fail = _max_row_id(tok_input['answer_coordinates'][0])  # first iteration that failed is when trying the original answer coordinates (max_row_id)
                        new_answer_cells = tok_input['answer_coordinates'][0]  # initialize variable with the last version tried (which are the original answer coordinates in iteration 0)
                    else:
                        min_fail = new_row_cutoff  # if failed decrease min_fail to the cutoff that was tried but resulted in an exception just now (shift window upper bound to left)
                    """
                    min_fail = _max_row_id(tok_input['answer_coordinates'][0])
                    if iteration == 0:
                        min_fail = _max_row_id(tok_input['answer_coordinates'][0])
                        max_sucess = None
                        max_row_id = _max_row_id(tok_input['answer_coordinates'][0])
                        all_answer_cells = [coordinate for coordinate in tok_input['answer_coordinates'][0]
                                            if coordinate[0] <= max_row_id
                                            ]
                        try:
                            tokenizer(**{key: value if key != 'answer_coordinates' else [all_answer_cells]
                                         for key, value in tokenizer_inputs.items()
                                         }
                                      )
                        except ValueError as e:
                            logger.debug(f"Testing modified answer coordinates {e}.")
                            all_answer_cells_fit = False
                    """
                new_answer_cells, new_row_cutoff = _compute_next_answer_coordinates(new_answer_cells, max_success, min_fail)  # update new answer cells and cutoff
                # when window is completely collapsed overwrite the answer coordinates and leave the loop
                if min_fail - max_success <= 1:
                    tok_input['answer_coordinates'] = [new_answer_cells]
                    break
                iteration += 1
