import json
from datetime import datetime
from pathlib import PurePath, Path
from typing import Dict, List, Callable, Optional, Union

import datasets

from numerical_table_questions.utils.data_caching import save_version


DUMMY_DATA_PATH = f"{__file__.replace(PurePath(__file__).name, '')}dummy_data_lm_eval_outout.jsonl"


def extract_jsonl_fields(file_path: str, fields: List[str] = ['table_idx', 'question_number', 'question', 'answer', 'aggregator', 'aggregation_num_rows'], metric_names: List[str] = ['exact_match'], save_path: Optional[str] = None) -> datasets.Dataset:
    model_name = PurePath(file_path).parent.name
    with open(file_path, 'r') as json_file:
        jsons = list(json_file)

    result_list = []
    for json_sample in jsons:
        data_dict = json.loads(json_sample)
        extracted_fields = {field: data_dict['doc'][field] for field in fields}
        extracted_fields.update({'doc_id': data_dict['doc_id'], 'resps': data_dict['resps'], 'filtered_resps': data_dict['filtered_resps'], 'model_name': model_name})
        extracted_fields.update({metric_name: data_dict[metric_name] for metric_name in metric_names})
        result_list.append(extracted_fields)
    results_dataset = datasets.Dataset.from_list(result_list)
    if save_path is not None:
        return save_version(results_dataset, save_path)
    return results_dataset


def add_table_length():
    # index_dataset = QuestionTableIndexDataset(table_question_dataset_path)
    #lm_results.map(lambda x: {'table_length': len(index_dataset[x['table_idx']]['data_dict']['rows']}), num_proc=12)
    pass


def question_to_main_expression(question: str) -> str:
    if 'of the difference between column' in question:
        return 'difference'
    if 'of the ratio of row' in question:
        return 'ratio'
    if 'of the expression' in question:
        return 'expression'
    return 'single_column'


def question_to_condition(question: str) -> str:
    if 'has a value equal to' in question:
        return '='
    if 'has a value different from' in question:
        return '!='
    if 'has a value greater than' in question:
        return '>'
    if 'has a value greater or equal than' in question:
        return '>='
    if 'has a value lesser than' in question:
        return '<'
    if 'has a value lesser or equal than' in question:
        return '<='
    return ''


def add_template_classes(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset = dataset.map(lambda x: {'main_expression': question_to_main_expression(x['question']),
                                     'condition': question_to_condition(x['question'])},
                          desc="Adding template class fields to dataset...",
                          num_proc=12)
    return dataset


def extract_all_from_dir(dir_path: str) -> datasets.Dataset:
    results_datasets = []
    for model_results in Path(dir_path).iterdir():
        if model_results.is_dir():
            for file in model_results.iterdir():
                if file.suffix == '.jsonl':
                    model_data = extract_jsonl_fields(str(file))
                    model_data = add_template_classes(model_data)
                    results_datasets.append(model_data)
    return datasets.concatenate_datasets(results_datasets)


def convert_jsonl_dataset_schema(file_path: str, rename_fields: Dict[str, str], remove_fields: Union[List[str], bool] = True , save_path: Optional[str] = None, num_proc: int = 12) -> datasets.Dataset:
    """ Rename and select/remove certain columns from the schema of a jsonl file. Cannot access nested fields. """
    jsonl_dataset = datasets.Dataset.from_list(file_path)
    if remove_fields is False:
        remove_fields = []
    elif remove_fields is True:
        remove_fields = list(jsonl_dataset.column_names)

    results_dataset = jsonl_dataset.map(lambda x: {new: x[old] for old, new in rename_fields.items()},
                                        remove_columns=list(set(remove_fields + list(rename_fields.keys()))),
                                        desc="Changing schema of original jsonl...",
                                        num_proc=num_proc,
                                        )
    if save_path is not None:
        save_version(results_dataset, save_path)
    return results_dataset


def calculate_posthoc_metrics(dataset: datasets.Dataset, metrics: Dict[str, Callable], fn_kwargs: dict = {}, num_proc: int = 12, save_path: Optional[str] = None, infer_save_path: bool = False) -> datasets.Dataset:
    new_dataset = dataset.map(lambda x: {metric_name: metric_func(x['filtered_resps'], [x['answer']])[0]
                                         for metric_name, metric_func in metrics.items()},
                              desc=f"Computing metrics {list(metrics.keys())} posthoc from responses...",
                              fn_kwargs=fn_kwargs,
                              num_proc=num_proc,
                              )
    if save_path is not None:
        save_version(new_dataset, save_path)
    elif infer_save_path:
        save_version(new_dataset, str(PurePath(dataset.cache_files[0]['filename']).parent.parent))
    return new_dataset


def calculate_results(file_path: str, distinguish_field: str = 'aggregator', distinguish_values: Optional[Union[List[str], List[float]]] = None, metric_names: List[str] = ['exact_match'], distinguish_samples: Optional[int] = None, num_bins: int = 10) -> dict:
    path_object = PurePath(file_path)
    results_path = file_path.replace(path_object.name, '') + f"/{distinguish_field}_results_{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}_" + path_object.name.replace('.jsonl', '.json')
    try:
        data = datasets.load_from_disk(file_path)
    except FileNotFoundError:
        data = extract_jsonl_fields(file_path, fields=[distinguish_field], metric_names=metric_names)
    data_len = len(data)
    if data_len == 0:
        return {}
    if distinguish_values is None:
        num_samples = distinguish_samples or data_len
        distinguish_values = list(set([data[int((i/num_samples)*data_len)][distinguish_field] for i in range(num_samples)]))
        try:
            last_bin = int(max([float(value) for value in distinguish_values]))
            distinguish_values = list(range(0, last_bin, int(last_bin//num_bins)))
            distinguish_values_is_range = True
        except ValueError:
            distinguish_values_is_range = False
    distingushed_metrics_dict = {distinguish_value: {metric_name: 0 for metric_name in metric_names} for distinguish_value in distinguish_values}
    value_count_dict = {distinguish_value: 0 for distinguish_value in distinguish_values}
    for sample in data:
        if distinguish_values_is_range:  # sample dict is the closest value
            absolute_differences = [abs(value - float(sample[distinguish_field])) for value in distinguish_values]
            argmin = min(range(len(absolute_differences)), key=lambda x : absolute_differences[x])
            nearest_neighbor = distinguish_values[argmin]
            sample_dict = distingushed_metrics_dict[nearest_neighbor]
            value_count_dict[nearest_neighbor] += 1
        else:  # sample dict is simply the distinguish_field value
            sample_dict = distingushed_metrics_dict[sample[distinguish_field]]
            value_count_dict[sample[distinguish_field]] += 1
        for metric_name in metric_names:
            sample_dict[metric_name] += sample[metric_name]
    # normalize metrics by the number of samples in each distinguish_value
    for distinguish_value in distingushed_metrics_dict.keys():
        for metric_name in metric_names:
            distingushed_metrics_dict[distinguish_value][metric_name] /= value_count_dict[distinguish_value] if value_count_dict[distinguish_value] != 0 else 1
        distingushed_metrics_dict[distinguish_value].update({'num_samples': value_count_dict[distinguish_value]})
    with open(results_path, 'w') as f:
        json.dump(distingushed_metrics_dict, f)
    return distingushed_metrics_dict
