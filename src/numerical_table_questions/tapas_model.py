import transformers


def tapas_model_type_info(model_name_or_path):
    return ModelTypeInfo(
        model_name_or_path=model_name_or_path,
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


def tapas_model_config(training_args):
    non_default_kwargs = dict()
    if training_args.model_name_or_path.lower() == 'tapas':
        model = transformers.TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
        # potentially change model config
        # model.config.xxx = 'xxx'
        non_default_kwargs['model_type_info'] = get_model_type_info(training_args.model_name_or_path)
        non_default_kwargs['generation_metrics'] = {
            'str_match_accuracy':  (str_match_accuracy,
                                    {
                                       'strip_whitespace': True,
                                    }
                                    )
        }
