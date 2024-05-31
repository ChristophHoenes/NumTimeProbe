import transformers

from numerical_table_questions.metrics import str_match_accuracy


def tapex_model_type_info() -> dict:
    return dict(
        model_name_or_path='tapex',
        pad_token_id=1,
        mask_token_id=-100,
        input_targets=True,
        loss_out_id='loss',
        filter_data_attributes=['input_ids', 'attention_mask'],
        input_mapping={
            '*': None,
            'labels': lambda x, y: y,
            },
        dict_input_mapping={
                    'input_ids': 'input_ids',
                    'attention_mask': 'attention_mask',
                    'labels': 'targets',
                    },
        )


def tapex_model():
    model = transformers.BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")
    # potentially change model config
    # model.config.xxx = 'xxx'
    return model


def tapex_config() -> dict:
    config = dict(
        generation_metrics={
            'str_match_accuracy': (str_match_accuracy,
                                   {
                                    'post_processing_fn': lambda x: [item.strip() for item in x],
                                    }
                                   )
            }
        )
    return config
