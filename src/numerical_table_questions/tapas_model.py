import transformers
from tqdm import tqdm

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
