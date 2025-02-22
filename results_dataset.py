import builtins
import json
import sys
import warnings
from datetime import datetime
from pathlib import PurePath, Path
from typing import Callable, Dict, List, Optional, Union

import datasets
import pandas as pd

from numerical_table_questions.data_loading import path_from_components
from numerical_table_questions.metrics import float_match_accuracy, absolute_distance
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset
from numerical_table_questions.utils.data_caching import save_version, caching
from numerical_table_questions.utils.plots import grouped_plot
from numerical_table_questions.utils.wandb_utils import get_artifact, get_run


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
    if 'of the ratio of column' in question:
        return 'ratio'
    if 'of the expression' in question:
        return 'expression'
    return 'single column'


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


def add_template_classes(dataset: datasets.Dataset, num_proc: int = 12) -> datasets.Dataset:
    dataset = dataset.map(lambda x: {'main_expression': question_to_main_expression(x['question']),
                                     'condition': question_to_condition(x['question'])},
                          desc="Adding template class fields to dataset...",
                          num_proc=num_proc)
    return dataset


def combine_datasets(dataset_list: List[datasets.Dataset], save_path: Optional[str] = None) -> datasets.Dataset:
    try:
        combined = datasets.concatenate_datasets([dataset for dataset in dataset_list])
    except ValueError:
        first_dataset = dataset_list[0]
        combined = first_dataset
        for dataset in dataset_list[1:]:
            combined = add_results(combined, dataset)
    if save_path is not None:
        save_version(combined, cache_path=save_path)
    return combined


def extract_all_from_dir(dir_path: str, metrics: Dict[str, Callable] = {'float_match': float_match_accuracy, 'absolute_distance': absolute_distance}, num_proc: int = 12, save_path: Optional[str] = None) -> datasets.Dataset:
    results_datasets = []
    for model_results in Path(dir_path).iterdir():
        if model_results.is_dir():
            model_id = model_results.name
            for file in model_results.iterdir():
                if file.suffix == '.jsonl':
                    model_data = extract_jsonl_fields(str(file))
                    model_data = add_template_classes(model_data, num_proc=num_proc)
                    metadata = extract_metadata(file)
                    model_data = model_data.map(lambda x: {'model_id': model_id, **metadata}, desc="Adding metadata...", num_proc=num_proc)
                    model_data = calculate_posthoc_metrics(model_data, metrics=metrics, num_proc=num_proc)
                    results_datasets.append(model_data)
    results_dataset = combine_datasets(results_datasets, save_path=save_path)
    return results_dataset


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
    """ Calculates metrics for every distinguish_value in distinguish_field of the dataset. Or creates a histogram if continous values.
        Kind of similar to create_bins but code is maybe harder to understand -> compare results.
    """
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


def get_suitable_dummy_value(value):
    match type(value):
        case builtins.str:
            return ''
        case builtins.int:
            return -1
        case builtins.float:
            return float('nan')
        case builtins.list:
            return []
        case builtins.dict:
            return {}
        case builtins.tuple:
            return tuple()
        case _:
            warnings.warn(f"No explicit dummy value implemented for type {type(value)}. Using None.")
            return None


def seq_to_item(sequence, recursion_depth=0, max_recursion_depth: int = 5):
    if recursion_depth >= max_recursion_depth:
        raise ValueError(f"Maximum recursion depth of {max_recursion_depth} is reached. Cannot flatten lists nested deeper than max_recursion_depth.")
    if isinstance(sequence, (list, tuple)):
        sequence = seq_to_item(sequence[0], recursion_depth=recursion_depth+1)
    return sequence


def flatten_nested_list_features(dataset: datasets.Dataset, fields: List[str], num_proc: int = 12, max_recursion_depth: int = 5) -> datasets.Dataset:
    dataset = dataset.map(lambda x: {field: seq_to_item(x[field], max_recursion_depth=max_recursion_depth) for field in fields},
                          desc="Flattening nested list features...",
                          num_proc=num_proc,
                          load_from_cache_file=False,  # recursion prevents hashing and causes warnings and performance decrease
                          )
    return dataset


def add_results(new_results: datasets.Dataset, old_dataset: datasets.Dataset, save_path: Optional[str] = None, infer_save_path: bool = False) -> datasets.Dataset:
    missing_columns = list(set(old_dataset.column_names) - set(new_results.column_names))
    if len(missing_columns) > 0:
        warnings.warn(f"Columns {missing_columns} are missing in the new_results! Adding empty values...")
        dummy_values = {col: get_suitable_dummy_value(old_dataset[0][col]) for col in missing_columns}
        new_results = new_results.map(lambda _: dummy_values)
    old_dataset = flatten_nested_list_features(old_dataset, fields=['filtered_resps', 'resps'])
    new_results = flatten_nested_list_features(new_results, fields=['filtered_resps', 'resps'])
    updated_dataset = datasets.concatenate_datasets([old_dataset, new_results])
    if save_path is None:
        if infer_save_path:
            if len(old_dataset.cache_files) == 0:
                raise ValueError("Cannot infer path from in-memory dataset (old_dataset does not have a cache file). Please provide a save_path.")
            cache_file_path = PurePath(old_dataset.cache_files[0]['filename'])
            #dataset_name = cache_file_path.parent.parent.name
            #cache_path = cache_file_path.parent.parent.parent
            save_path = cache_file_path.parent.parent
            save_version(updated_dataset, cache_path=save_path)
    else:
        save_version(updated_dataset, cache_path=save_path)
    return updated_dataset


def results_datasets_from_wandb(wandb_run_id: str,
                                results_table_name: str = 'results',
                                dataset_fields: List[str] = ['table_idx', 'question_number', 'question_id', 'question', 'answer', 'aggregator', 'aggregation_num_rows'],
                                metric_name_map: Dict[str, str] = {'exact_match': 'str_match_accuracy', 'float_match': 'float_match_accuracy', 'absolute_distance': 'absolute_distance'},
                                save_path: Optional[str] = None,
                                infer_save_path: bool = False,
                                ) -> List[datasets.Dataset]:
    wandb_run = get_run(wandb_run_id)
    run_arguments = wandb_run.config
    results_table = get_artifact(wandb_run_id, artifact_name=results_table_name, return_local_path=True)
    # TODO get underlying dataset and extract fields
    # fields: List[str] = ['table_idx', 'question_number', 'answer', 'aggregator', 'aggregation_num_rows']
    #try:
    #    local_artifact = get_artifact(wandb_run_id, artifact_name="eval_dataset_path", return_local_path=True)
    #except Exception as e:
    #    raise e
    # TODO check what happens to latest version vs. run version is caching used?
    # for runs on KISZ node the training args are saved as string representation of the TrainingArgs class -> need to parse them
    if isinstance(run_arguments['training_args'], str):
        run_arguments['training_args'] = eval(run_arguments['training_args'].replace('TrainingArgs', 'dict'))
    dataset_path = path_from_components(run_arguments['training_args']['table_corpus_name'], run_arguments['training_args']['dataset_suffix'], 'test', data_dir=run_arguments['training_args']['data_dir'])
    artifact_shard_datasets = []
    for artifact_shard_id, file_path in enumerate(results_table):
        with open(file_path, 'r') as file:
            data_dict = json.load(file)
            col_to_id_mapping = {col: idx for idx, col in enumerate(data_dict['columns'])}
            extracted_dicts = {}
            for data in data_dict['data']:
                wandb_fields = {'doc_id': data[col_to_id_mapping['question_ids']],
                                'resps': data[col_to_id_mapping['text_predictions']],
                                'filtered_resps': data[col_to_id_mapping['processed_text_predictions']],
                                'model_name': run_arguments['training_args']['model_name_or_path'],
                                'artifact_shard_id': artifact_shard_id,
                                }
                # add all metrics specified in metrics map
                wandb_fields.update({col_name: data[col_to_id_mapping[metric_name]] for col_name, metric_name in metric_name_map.items()})
                extracted_dicts[wandb_fields['doc_id']] = wandb_fields
            for dataset_sample in QuestionTableIndexDataset(dataset_path, lm_eval_style=True):
                dataset_fields = {field: dataset_sample[field] for field in dataset_fields}
                coresponding_results = extracted_dicts.get(dataset_sample['question_id'])
                if coresponding_results is not None:
                    coresponding_results.update(dataset_fields)
                else:
                    warnings.warn(f"Could not find results for question_id {dataset_sample['question_id']} in the extracted results.")
            artifact_shard_dataset = datasets.Dataset.from_list(list(extracted_dicts.values()))
            artifact_shard_datasets.append(artifact_shard_dataset)
            if save_path is not None:
                save_version(artifact_shard_dataset, save_path)
            elif infer_save_path:
                save_version(artifact_shard_dataset, f"results_dataset_{wandb_run_id}" + (f"_shard_{artifact_shard_id}" if artifact_shard_id > 0 else ''))
    return artifact_shard_datasets


def grouped_results(dataset: datasets.Dataset, group_by_cols: List[str], row_filters: Dict[str, Callable] = {}, select_cols: List[str] = ['exact_match', 'float_match', 'absolute_distance']) -> datasets.Dataset:
    if len(row_filters) > 0:
        dataset = dataset.filter(lambda x: all([condition(x[col]) for col, condition in row_filters.items()]), num_proc=12)
    df = dataset.to_pandas()
    df = df[group_by_cols + select_cols]
    if len(group_by_cols) > 1:
        #arrays = [df[col].values for col in group_by_cols]
        #index = pd.MultiIndex.from_arrays(arrays, names=group_by_cols)
        df.set_index(group_by_cols, inplace=True)
    #df = df.groupby(level=group_by_cols)[select_cols].mean()
    aggregations = {col: ['mean', 'count'] for col in select_cols}
    df = df.groupby(level=group_by_cols).agg(aggregations)
    return df


# filter the one invalid sample from the dataset lambda x: int(x['aggregation_num_rows']) > 0
def filter_results(dataset: datasets.Dataset, filters: Dict[str, Callable]) -> datasets.Dataset:
    return dataset.filter(lambda x: all([condition(x[col]) for col, condition in filters.items()]), num_proc=12)


def decide_bin(sample, bin_col: str, ranges: Union[List[int], List[float]]):
    for i, boundary in enumerate(ranges):
        if float(sample[bin_col]) >= boundary:
            last_boundary = (str(i), boundary)
            continue
    # string boundary leads to plot formating issues
    # 'boundary': f"{ranges[last_boundary[0]+1]}-{ranges[last_boundary[0]+1]}" if last_boundary[0] < len(ranges) - 1 else f">{ranges[last_boundary[0]]}"
    return {'bin': last_boundary[0], 'boundary': last_boundary[1]}


# good value for bin_ranges=[0,3,10,25,50] on wikitablequestions
def create_bins(dataset: datasets.Dataset, bin_col: str, num_bins: int = 5, bin_ranges: Optional[Union[List[int], List[float]]] = None, mode: str = 'equal_count') -> datasets.Dataset:
    if bin_ranges is None:
        if mode == 'equal_step':
            min_val = min([float(val) for val in dataset[bin_col]])
            max_val = max([float(val) for val in dataset[bin_col]])
            step = int((max_val - min_val) / num_bins)
            bin_ranges = list(range(int(min_val), int(max_val), step))
        elif mode == 'equal_count':
            sorted_values = list(sorted([float(val) for val in dataset[bin_col]]))
            step = len(sorted_values) // num_bins
            bin_ranges = []
            for i in range(0, len(sorted_values), step):
                # if a value occurs so often that it would span more than one bin search for the next unique value
                if sorted_values[i] not in bin_ranges:
                    bin_ranges.append(sorted_values[i])
                else:
                    found_unique = False
                    for j in range(i, len(sorted_values)):
                        if sorted_values[j] not in bin_ranges:
                            bin_ranges.append(sorted_values[i])
                            found_unique = True
                            break
                    if not found_unique:
                        raise ValueError(f"Could not find unique value for bin {len(bin_ranges)}. Not enough unique values.")
        else:
            raise ValueError(f"Mode {mode} is not implemented. Please use 'equal_step' or 'equal_count'.")
    dataset = dataset.map(decide_bin, fn_kwargs={'bin_col': bin_col, 'ranges': bin_ranges}, desc="Deciding bins...", num_proc=12)
    return dataset


def parse_axes_labels(x_col: Optional[str] = None, y_col: Optional[str] = None):
    axes_args = {}

    def _match_column(col_name: str):
        match col_name:
            case s if 'aggregation_num_rows' in s:
                return 'Number of rows before aggregation'
            case s if 'exact_match' in s:
                return 'Exact match accuracy'
            case s if 'float_match' in s:
                return 'Float match accuracy'
            case s if 'absolute_distance' in s:
                return 'Average Relative Distance Truncated @ 10'
            case s if 'aggregator' in s:
                return 'Aggregator'
            case s if 'main_expression' in s:
                return 'Main Expression Type'
            case s if 'condition' in s:
                return 'Condition Type'
            case _:
                pass

    if x_col is not None:
        axes_args.update({'xlabel': _match_column(x_col)})
        #axes_args.update({'xticklabels': _process_xtick_labels(x_col, data)})
    if y_col is not None:
        axes_args.update({'ylabel': _match_column(y_col)})
    return axes_args


def dat_to_line(dat, x_col='aggregation_num_rows', y_col='float_match_mean', row_filters={'aggregation_num_rows': lambda x: int(x) > 0}, num_bins=5, bin_mode='equal_count', bin_ranges=[0, 3, 10, 50, 100]):
    dat_bin = create_bins(dat, bin_col=x_col, num_bins=num_bins, mode=bin_mode, bin_ranges=bin_ranges)
    df = grouped_results(dat_bin, group_by_cols=['model_name', 'boundary'], row_filters=row_filters, select_cols=[y_col.replace('_mean', '').replace('_count', '')])
    df_long_form = df.reset_index()
    df_long_form.columns = ['_'.join([c for c in col if c != '']) for col in df_long_form.columns]
    axes_args = parse_axes_labels(x_col=x_col, y_col=y_col)
    grouped_plot(df_long_form, x='boundary', y=y_col, kind='line', axes_args=axes_args)


def dat_to_bar(dat, category='aggregator', y_col='float_match_mean', row_filters={'aggregation_num_rows': lambda x: int(x) > 0}):
    df = grouped_results(dat, group_by_cols=['model_name', category], row_filters=row_filters, select_cols=[y_col.replace('_mean', '').replace('_count', '')])
    df_long_form = df.reset_index()
    df_long_form.columns = ['_'.join([c for c in col if c != '']) for col in df_long_form.columns]
    axes_args = parse_axes_labels(x_col=category, y_col=y_col)
    grouped_plot(df_long_form, x=category, y=y_col, kind='bar', axes_args=axes_args)


def dat_to_hist(dat, x_col='main_expression', row_filters={'aggregation_num_rows': lambda x: int(x) > 0}):
    df = grouped_results(dat, group_by_cols=['model_name', x_col], row_filters=row_filters, select_cols=['doc_id'])
    df_long_form = df.reset_index()
    axes_args = parse_axes_labels(x_col=x_col)
    grouped_plot(df_long_form, x=x_col, kind='hist', axes_args=axes_args)


def debug_expression_answers(dat_expr):
    hist = {}
    for i in range(len(dat_expr)):
        if hist.get(dat_expr[i]['answer']) is None:
            hist[dat_expr[i]['answer']] = 1
        else:
            hist[dat_expr[i]['answer']] += 1
    print(hist)


def standard_plots(dataset: datasets.Dataset):
    dat_to_bar(dataset, category='aggregator')
    dat_to_bar(dataset, category='main_expression')
    dat_to_bar(dataset, category='condition')
    dat_to_line(dataset)


def extract_metadata(jsonl_file: PurePath) -> dict:
    model_id = str(jsonl_file.parent.name)
    time_stamp = jsonl_file.name.split('_')[-1].strip('.jsonl').replace('.', '_')
    save_path = PurePath(f'{lm_eval_results_path}/{model_id}_{time_stamp}_results_dataset')
    results_path = (jsonl_file.parent / f"results_{time_stamp}.json")
    if results_path.exists():
        with open(results_path, 'r') as results_path:
            results_json = json.load(results_path)
        system_instruction = results_json['system_instruction']
        chat_template = results_json['chat_template']
        num_fewshot = results_json['configs']['num_tab_qa_gittables_100k']['num_fewshot']
        fewshot_as_multiturn = results_json['fewshot_as_multiturn']
        max_length = results_json['max_length']
        eval_time = results_json['total_evaluation_time_seconds']
        metadata = {'model_id': model_id, 'system_instruction': system_instruction, 'chat_template': chat_template,
                    'num_fewshot': num_fewshot, 'fewshot_as_multiturn': fewshot_as_multiturn, 'max_length': max_length, 'eval_time': eval_time
                    }
        with open(save_path.parent / model_id / 'metadata.json', 'w+') as f:
            json.dump(metadata, f)
    else:
        metadata = {}
    return metadata


if __name__ == '__main__':
    arguments = sys.argv[1:]
    lm_eval_results_path = arguments[0]
    wandb_run_ids = arguments[1:]
    cache_path = '/home/mamba/.cache'

    # lm_eval results
    #lm_eval_results_path = '/home/mamba/.cache/results/wikitablequestions/250209'
    #dummy_path = '/home/mamba/.cache/results/dummy'
    #lm_eval_results_path = dummy_path
    lm_eval_combined_name = 'combined'
    lm_eval_combined_path = Path(lm_eval_results_path) / lm_eval_combined_name
    if lm_eval_combined_path.is_dir():
        lm_eval_results = caching(str(lm_eval_combined_path.name), cache_path=str(lm_eval_combined_path.parent))
    else:
        lm_eval_results = extract_all_from_dir(lm_eval_results_path, save_path=str(lm_eval_combined_path))
    standard_plots(lm_eval_results)

    # wandb datasets
    if Path(wandb_run_ids[0]).is_dir():
        wandb_run_datasets = [caching(str(PurePath(wandb_run_ids[i]).name), cache_path=str(PurePath(wandb_run_ids[i]).parent)) for i in range(len(wandb_run_ids))]
    else:
        wandb_run_datasets = []
        for run_id in wandb_run_ids:
            wandb_dataset_shards = results_datasets_from_wandb(run_id, save_path=str(PurePath(lm_eval_results_path).parent / 'wandb' / run_id / 'shards'))
            run_dataset = combine_datasets(wandb_dataset_shards, save_path=str(PurePath(lm_eval_results_path).parent / 'wandb' / run_id))
            wandb_run_datasets.append(run_dataset)

    # if wandb_results available first join them among each other and then join them with lm_eval results
    if len(wandb_run_datasets) > 0:
        if len(wandb_run_datasets) > 1:
            wandb_results = combine_datasets(wandb_run_datasets, save_path=str(PurePath(lm_eval_results_path).parent / 'wandb' / 'combined'))
            standard_plots(wandb_results)
        else:
            wandb_results = wandb_run_datasets[0]
        # all results combined
        all_combined_name = 'combined'
        all_combined_path = Path(lm_eval_results_path).parent / all_combined_name
        if all_combined_path.is_dir():
            combined_results = caching(str(all_combined_path.name), cache_path=str(all_combined_path.parent))
        else:
            combined_results = combine_datasets([lm_eval_results, wandb_results], save_path=str(all_combined_path))
        standard_plots(combined_results)
    else:
        combined_results = lm_eval_results

    grouped_df = grouped_results(combined_results, group_by_cols=['model_id', 'aggregator'], row_filters={})
    print(grouped_df)
