import builtins
import warnings
import json
from pathlib import PurePath, Path
from typing import Dict, List, Optional, Callable

import datasets
import pandas as pd

from numerical_table_questions.data_loading import path_from_components
from numerical_table_questions.extract_lm_eval_data import extract_jsonl_fields, calculate_posthoc_metrics
from numerical_table_questions.metrics import float_match_accuracy, absolute_distance
from numerical_table_questions.lazy_data_processing import QuestionTableIndexDataset
from numerical_table_questions.utils.data_caching import save_version, caching
from numerical_table_questions.utils.wandb_utils import get_artifact, get_run


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


def add_results(new_results: datasets.Dataset, old_dataset: datasets.Dataset, save_path: Optional[str] = None, infer_save_path: bool = False) -> datasets.Dataset:
    missing_columns = list(set(old_dataset.column_names) - set(new_results.column_names))
    if len(missing_columns) > 0:
        warnings.warn(f"Columns {missing_columns} are missing in the new_results! Adding empty values...")
        dummy_values = {col: get_suitable_dummy_value(old_dataset[0][col]) for col in missing_columns}
        new_results = new_results.map(lambda _: dummy_values)
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
                               dataset_fields: List[str] = ['table_idx', 'question_number', 'answer', 'aggregator', 'aggregation_num_rows'],
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


def grouped_results(dataset: datasets.Dataset, group_by_cols: List[str], row_filters: Dict[str, Callable], select_cols: List[str] = ['exact_match', 'float_match', 'absolute_distance']) -> datasets.Dataset:
    if len(row_filters) > 0:
        dataset = dataset.filter(lambda x: all([condition(x[col]) for col, condition in row_filters.items()]), num_proc=12)
    df = dataset.to_pandas()
    df = df[group_by_cols + select_cols]
    if len(group_by_cols) > 1:
        #arrays = [df[col].values for col in group_by_cols]
        #index = pd.MultiIndex.from_arrays(arrays, names=group_by_cols)
        df.set_index(group_by_cols, inplace=True)
    df = df.groupby(level=group_by_cols)[select_cols].mean()
    return df


if __name__ == '__main__':
    # lm_eval results
    lm_eval_results_path = '/home/mamba/.cache/results/wikitablequestions/250209'
    dummy_path = '/home/mamba/.cache/results/dummy'
    lm_eval_results_path = dummy_path
    is_first_dataset = True
    for dir in Path(lm_eval_results_path).iterdir():
        if dir.is_dir():
            model_id = dir.name
            for file in dir.iterdir():
                if file.name.endswith('.jsonl'):
                    time_stamp = file.name.split('_')[-1].strip('.jsonl').replace('.', '_')
                    save_path = PurePath(f'{lm_eval_results_path}/{model_id}_{time_stamp}_results_dataset')
                    results_path = (file.parent / f"results_{time_stamp}.json")
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
                    results_dataset = extract_jsonl_fields(str(file))
                    #results_dataset = caching(save_path.name, cache_path=str(save_path.parent))
                    results_dataset = results_dataset.map(lambda x: {'model_id': model_id, **metadata, **x}, desc="Adding metadata...", num_proc=12)
                    results_dataset = calculate_posthoc_metrics(results_dataset, metrics={'float_match': float_match_accuracy, 'absolute_distance': absolute_distance}, num_proc=12)
                    if is_first_dataset:
                        extended_results = results_dataset
                        is_first_dataset = False
                    else:
                        extended_results = add_results(extended_results, results_dataset)

    # wandb datasets
    wandb_run_ids = []
    for run_id in wandb_run_ids:
        wandb_dataset_shards = results_datasets_from_wandb(run_id)
        for wandb_dataset_shard in wandb_dataset_shards:
            extended_results = add_results(extended_results, wandb_dataset_shard)

    save_version(extended_results, cache_path=lm_eval_results_path, cache_file_name='joined_results_dataset')

    grouped_df = grouped_results(extended_results, group_by_cols=['model_id', 'aggregator'], row_filters={})
    print(grouped_df)
