import math
import transformers
import warnings
from tqdm import tqdm
from typing import List, Union, Tuple

import pandas as pd

from numerical_table_questions.answer_coordinates import AnswerCoordinates, compute_answer_coordinates
from numerical_table_questions.data_synthesis.table import Table, TableQuestionDataSet
from numerical_table_questions.metrics import str_match_accuracy


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
        dict_input_mapping ={},  # by default use all tokenizer keys as input
        )


def tapas_model():
    model = transformers.TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
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
                else:
                    answer_text = None
                    answer_coordinates = None
                processed_samples.append({
                    'table': table_df,
                    'queries': [question],
                    'answer_coordinates': [answer_coordinates],
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


def tapas_generation(tokenizer, model_inputs, model_outputs, table) -> List[str]:
    answer_coordinates, aggregation_indices = (
        tokenizer.convert_logits_to_predictions({key: value.detach().cpu()for key, value in model_inputs.items()},  # assumes dict input
                                                model_outputs['logits'].detach().cpu(),  # assumes dict type for model_outputs not BatchEncoding
                                                model_outputs['logits_aggregation'].detach().cpu()  # assumes dict type for model_outputs not BatchEncoding
                                                )
        )
    aggregators = get_aggregator_string(aggregation_indices)
    answer_cells = get_answer_cell_values(table, answer_coordinates)

    predictions = compute_aggregation(aggregators, answer_cells)
    return predictions


def reduce_answer_coordinates(tok_input: dict, iteration: int = 0, max_length=512):
    if iteration > max_length + 1:
        raise StopIteration("Iteration exceeded max_length of model. Aborting due to risk of infinity loop.")

    def _max_row_id(answer_coordinates: List[Tuple[int, int]]) -> int:
        return max([coordinate[0] for coordinate in answer_coordinates])

    max_row_id = _max_row_id(tok_input['answer_coordinates'][0])

    # in first iteration remove all cells that exceed the max_length of the model
    if iteration == 0:
        reduced_coordinates = [coordinate for coordinate in tok_input['answer_coordinates'][0]
                               if coordinate[0] * coordinate[1] <= max_length
                               ]
        # if the previous step did reduce the answer coordinates return result else proceed with removing last row
        if max_row_id != _max_row_id(reduced_coordinates):
            tok_input['answer_coordinates'][0] = reduced_coordinates

    # remove the row with highest index (max_row_id)
    tok_input['answer_coordinates'][0] = [coordinate for coordinate in tok_input['answer_coordinates'][0]
                                          if coordinate[0] < max_row_id
                                          ]
