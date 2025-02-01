import datasets
from numerical_table_questions.data_utils import consistent_quoting_map
from numerical_table_questions.data_utils import extract_column_names_from_table


dat = datasets.Dataset.load_from_disk('/home/mamba/.cache/gittables_group_filtered_standard_templates_test_quality_20k/250128_1032_17_123456')
dat_tab = datasets.Dataset.load_from_disk('/home/mamba/.cache/gittables_group_filtered_test_tables/241002_1315_37_395441')
table_column_map = extract_column_names_from_table(dat_tab)
dat_quote = dat.map(consistent_quoting_map, fn_kwargs={'table_column_names_map': table_column_map}, num_proc=1, load_from_cache_file=False)
