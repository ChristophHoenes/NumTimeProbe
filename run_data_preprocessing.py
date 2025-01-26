from dargparser import dargparse

from numerical_table_questions.arguments import TrainingArgs, MiscArgs, TokenizationArgs
from numerical_table_questions.data_loading import TableQADataModule
from numerical_table_questions.utils.model_utils import get_model_type_info


def main(parsed_arg_groups):
    args, misc_args, tokenizer_args = parsed_arg_groups
    dm = TableQADataModule(get_model_type_info(args.model_name_or_path),
                           table_corpus=args.table_corpus_name,
                           dataset_name=args.dataset_suffix,
                           train_batch_size=args.batch_size_per_device,
                           eval_batch_size=args.eval_batch_size_per_device,
                           tokenizing_args=tokenizer_args,
                           lazy_data_processing=args.lazy_data_processing,
                           is_batch_dict=args.is_batch_dict,
                           num_dataloader_workers=args.workers,
                           too_many_open_files_fix=misc_args.too_many_open_files_fix,
                           )
    dm.prepare_data()


if __name__ == "__main__":
    parsed_arg_groups = dargparse(dataclasses=(TrainingArgs, MiscArgs, TokenizationArgs))
    # TODO logger
    print(parsed_arg_groups)
    main(parsed_arg_groups)
