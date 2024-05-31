import transformers

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
