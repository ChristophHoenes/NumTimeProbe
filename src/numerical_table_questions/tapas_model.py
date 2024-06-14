import transformers
from tqdm import tqdm
from typing import List

import pandas as pd

from numerical_table_questions.answer_coordinates import AnswerCoordinates
from numerical_table_questions.data_synthesis import Table, TableQuestionDataSet
from numerical_table_questions.metrics import str_match_accuracy


def tapas_model_type_info() -> dict:
    # TODO check correct values
    return dict(
        model_name_or_path='tapas',
        pad_token_id=1,  # ?
        mask_token_id=-100,  # ?
        input_targets=True,
        loss_out_id='loss',
        filter_data_attributes=['input_ids', 'attention_mask'],
        input_mapping={
            '*': None,
            'labels': lambda x, y: y,
            }
        )


def tapas_model():
    model = transformers.TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
    # potentially change model config
    # model.config.xxx = 'xxx'
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
            for sample in (progress_bar := tqdm(data)):
                progress_bar.set_description("transfer to TAPAS tokenizer format...")
                table = Table.from_state_dict(sample['table'])
                table_df = table.pandas_dataframe
                # retrieve question
                question = sample['questions'][kwargs['question_number']]
                if not is_eval:  # for training Answer coordinates are required
                    answer_text = [sample['answers'][kwargs['question_number']]]
                    answer_coordinates = AnswerCoordinates(**sample['answer_coordinates'][kwargs['question_number']]).generate()
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
                    raise ValueError("Encountered multi-row answer with aggregator NONE (NOOP)! Make sure the answer is a single value.")
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
