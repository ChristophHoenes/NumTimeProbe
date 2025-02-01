import datasets
from numerical_table_questions.data_utils import gather_statistics_map, aggregate_statistics_map

dat = datasets.Dataset.load_from_disk('/home/mamba/.cache/gittables_group_filtered_standard_templates_test_quality_20k_quotes/250201_1539_11_123456')

stats_only = dat.map(gather_statistics_map, remove_columns=dat.column_names, num_proc=1)

stats_only_agg = aggregate_statistics_map(stats_only)
