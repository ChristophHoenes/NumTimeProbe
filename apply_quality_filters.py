import sys

import datasets

from numerical_table_questions.data_synthesis.template_creation import apply_quality_filters
from numerical_table_questions.utils.data_utils import consistent_quoting_map, cutoff_num_questions
from numerical_table_questions.utils.data_caching import caching, save_version


if __name__ == '__main__':
    dataset_names = sys.argv[1:]
    for dataset_name in dataset_names:
        dat = caching(dataset_name, cache_path='/home/mamba/.cache')
        dat_filtered = apply_quality_filters(dat, save=True, dataset_name=dataset_name, cache_path='/home/mamba/.cache')

        dat_strict = caching(dataset_name + '_filtered_multi_answer_filter_agg_count_0', cache_path='/home/mamba/.cache')

        if isinstance(dat_strict[0]['table'], str):
            tab_dat = datasets.Dataset.load_from_disk(dat_strict[0]['table_dataset_path'])
            table_column_names_map = {table['table']['table_id']: table['table']['deduplicated_column_names'] for table in tab_dat}
        else:
            table_column_names_map = None
        correct_quoting = dat_strict.map(consistent_quoting_map, fn_kwargs={'table_column_names_map': table_column_names_map}, num_proc=8, desc=f"Applying consistent quoting to {dataset_name}...")
        save_version(correct_quoting, cache_path='/home/mamba/.cache', cache_file_name=dataset_name + '_filtered_multi_answer_filter_agg_count_0_correct_quoting')

        if 'wikitable' in dataset_name:
            if 'train' in dataset_name:
                cutoff = 200_999  # 201474 - 1 is limit before one question more per table
            else:
                cutoff = 20_273
        else:
            # TODO rethink cutoffs
            if 'train' in dataset_name:
                cutoff = 270_138  # 270_137 is exactly two questions per table
            else:
                cutoff = 20_273  # 16932 validation tables 16931 test
        subsampled_dataset = cutoff_num_questions(correct_quoting, cutoff=cutoff, num_proc=8)
        for i in range(len(subsampled_dataset[0]['questions'])):
            if 'value' not in subsampled_dataset[0]['questions'][i]:
                print(subsampled_dataset[0]['questions'][i])
        save_version(subsampled_dataset, cache_path='/home/mamba/.cache', cache_file_name=dataset_name + f'_filtered_multi_answer_filter_agg_count_0_correct_quoting_{len(subsampled_dataset)}')
