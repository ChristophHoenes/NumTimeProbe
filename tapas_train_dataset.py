import datasets

from numerical_table_questions.data_synthesis.template_creation import apply_filter_condition
from results_dataset import question_to_main_expression


#dataset = datasets.Dataset.load_from_disk('/home/mamba/.cache/wikitablequestions_standard_templates_with_count_train_filtered_multi_answer_filter_agg_count_0_correct_quoting_861/250205_2119_12_444451')
#dataset = datasets.Dataset.load_from_disk('/home/mamba/.cache/wikitablequestions_standard_templates_with_count_train/250113_1645_08_387183')
#dataset = datasets.Dataset.load_from_disk('/home/mamba/.cache/wikitablequestions_standard_templates_with_count_validation_filtered_multi_answer_filter_agg_count_0_correct_quoting_216/250205_2117_27_865416')
dataset = datasets.Dataset.load_from_disk('/home/mamba/.cache/wikitablequestions_standard_templates_with_count_validation/250113_1936_05_160758')


dataset = dataset.map(
    lambda x: {'main_expression': [question_to_main_expression(question) for question in x['questions']]},
    desc="Adding main_expression field to dataset...",
    num_proc=12,
    )

dataset = dataset.map(
    lambda x: {'filter_condition': [main_expression == 'single_column' for main_expression in x['main_expression']]},
    desc="Prepare filter_condition: main_expression == single_column...",
    num_proc=12,
)
dataset = apply_filter_condition(dataset, num_proc=12)
dataset = dataset.remove_columns(['main_expression'])
