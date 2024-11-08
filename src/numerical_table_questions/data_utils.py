import random
import re
import tracemalloc
import wandb
from dargparser import dargparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import torch
import numpy as np
import pandas as pd

from numerical_table_questions.arguments import DataProcessingArgs
from numerical_table_questions.dlib.frameworks.wandb import WANDB_ENTITY, WANDB_PROJECT
#from src.numerical_table_questions.data_caching import caching
from numerical_table_questions.data_caching import caching, delete_dataset
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.table_creation import remove_duplicate_qa_pairs


DUMMY_DATA = datasets.Dataset.from_dict({
    'questions': ["What is the maximum of column revenue given that cost has value 1$?",
                  "What is the value of column cost given that name has value \t Mr. Prêsident\t?",
                  "What is the average of column some\nscore\t given that revenue has value 1132.346?",
                  ],
    'table': [dict(), dict(), dict()],
    'answer': ['13.67', '15$', '3']
})


def cast_to_reduced_int(ints: torch.Tensor, num_values: Optional[int] = None):
    """
        Selects the smallest possible torch dtype for ints representing an id mapping of size num_value.
        If num_values is None the amount of values (e.g. vocab size) is estimated by the maximum of the
        values in the tensor plus one (for id zero).
    """
    # if num_values is None infer the coding size
    if num_values is None:
        num_values = ints.max() + 1
    if num_values <= 2:
        cast_to = torch.bool
    elif num_values <= 128:
        cast_to = torch.int8
    elif num_values <= 256 and ints.min() >= 0:
        cast_to = torch.uint8
    elif num_values <= 32768:
        cast_to = torch.int16
    elif num_values <= 2_147_483_648:
        cast_to = torch.int32
    else:
        cast_to = torch.int64
    return ints.to(cast_to)


def find_enclosed_substring(search_string: str, prefix: str, postfix: str, normalize: bool = False):
    """ Searches for substring that is between a certain prefix and postfix."""
    if normalize:
        normalized_search_string = search_string.lower().strip()
        pre = prefix.lower()
        post = postfix.lower()
    else:
        normalized_search_string = search_string
        pre = prefix
        post = postfix

    result = re.search(fr"(?P<arbitrary_prefix>(.*))(?P<prefix>{pre})(?P<target>(.*))(?P<postfix>{post})(?P<arbitrary_postfix>(.*))", repr(normalized_search_string))

    if result is not None:
        return result.group('target')
    else:
        return ''


def extract_aggregator_from_text(dataset_mapping: dict, source_column_name: str, target_column_name: str, search_beginning: str, search_end: str):
    text_column = dataset_mapping[source_column_name]
    if isinstance(text_column, (list, tuple)):
        value = [find_enclosed_substring(text, search_beginning, search_end) for text in text_column]
    else:
        value = [find_enclosed_substring(text_column, search_beginning, search_end)]
    aggregators = []
    for position_of_interest in value:
        match position_of_interest.lower().strip():
            case 'maximum' | 'max' | 'highest' | 'greatest':
                aggregators.append('max')
            case 'minimum' | 'min' | 'lowest' | 'smallest':
                aggregators.append('min')
            case 'sum':
                aggregators.append('sum')
            case 'average' | 'avg' | 'mean':
                aggregators.append('avg')
            case 'value':
                aggregators.append('noop')
            case _:
                aggregators.append('unknown')
    if len(aggregators) == 1:
        return {target_column_name: aggregators[0]}
    else:
        return {target_column_name: aggregators}


def extract_field_from_text(dataset_mapping: dict,
                            source_column_name: str,
                            target_column_name: str,
                            search_beginning: str,
                            search_end: str):
    """ Searches for substring (between case-sensitive search_beginning and search_end patterns) in a text column and extracts it as a new field in the dataset.
        Special regex characters need to be escaped in search_beginning and search_end, respectively
    """
    text_column = dataset_mapping[source_column_name]
    if isinstance(text_column, (list, tuple)):
        value = [find_enclosed_substring(text, search_beginning, search_end) for text in text_column]
    else:
        value = find_enclosed_substring(text_column, search_beginning, search_end)
    return {target_column_name: value}


def extract_template_fields_from_query(dataset):
    # extract aggregator
    search_prefix = "What is the "
    search_postfix = " of column"
    source_column_name = "questions"
    dataset, _ = dataset.map(extract_aggregator_from_text,
                             fn_kwargs={
                                 'source_column_name': source_column_name,
                                 'target_column_name': 'aggregators',
                                 'search_beginning': search_prefix,
                                 'search_end': search_postfix
                                 }
                             ), delete_dataset(dataset)
    # extract column name
    search_prefix = " of column "
    search_postfix = " given that"
    dataset, _ = dataset.map(extract_field_from_text,
                             fn_kwargs={
                                 'source_column_name': source_column_name,
                                 'target_column_name': 'column_names',
                                 'search_beginning': search_prefix,
                                 'search_end': search_postfix
                                 }
                             ), delete_dataset(dataset)
    # extract condition value
    search_prefix = "given that "
    search_postfix = r"\?"  # escape because ? is special character in regex
    dataset, _ = dataset.map(extract_field_from_text,
                             fn_kwargs={
                                 'source_column_name': source_column_name,
                                 'target_column_name': 'condition_values',
                                 'search_beginning': search_prefix,
                                 'search_end': search_postfix
                                 }
                             ), delete_dataset(dataset)
    return dataset


def extract_properties_posthoc(args, use_dummy_data=False):
    if use_dummy_data:
        data_split = DUMMY_DATA
    else:
        base_filename = f"{args.table_corpus}_{args.split}_{args.dataset_name}"
        data_split = caching(base_filename, cache_path=args.data_dir)

    deduplicated = data_split.map(remove_duplicate_qa_pairs)
    output = extract_template_fields_from_query(deduplicated)
    output.save_to_disk(Path(args.data_dir) / ("stats_" + base_filename)), delete_dataset(output)
    return output


def load_artifact_from_wandb(run_id, artifact_name='text_predictions', version='latest', wandb_entity=WANDB_ENTITY, wandb_project=WANDB_PROJECT):
    api = wandb.Api()
    artifact = api.artifact(f"{wandb_entity}/{wandb_project}/run-{run_id}-{artifact_name}:{version}")
    return artifact.get("text_predictions")


def add_column_from_artifact(dataset, artifact, column_name: Union[str, List[str], Dict[str, str]]):
    # parse name mapping to uniform format (dict)
    if isinstance(column_name, str):
        column_name_mapping = {column_name: column_name}  # keep name of single column
    elif isinstance(column_name, list):
        column_name_mapping = dict(zip(column_name, column_name))  # keep name of every column
    else:
        column_name_mapping = column_name  # explicit name maping from dataset name to artifact name provided by user

    # add specified columns of artifact
    artifact_df = artifact.get_dataframe()
    for name_table, name_artifact in column_name_mapping.items():
        dataset = dataset.add_column(name_table, artifact_df[name_artifact].to_list())
    return dataset


def extract_table_id(dataset):
    return {'table_id': dataset['table']['table_id']}


def is_prediction_correct(data_sample, comparison_fn=...):
    return {'is_correct': comparison_fn(data_sample['text_prediction'], data_sample['answers'])}


# TODO use this as test if sampled values correspond to the specified condition column for basic template
def is_value_in_condition_col(data_sample):
    df = Table.from_state_dict(data_sample['table']).pandas_dataframe
    return {'is_value_in_condition_col': data_sample['value'] in df.get(data_sample['condition_col'], default=pd.Series()).unique().tolist()}


def sample_questions(table_dataset_sample: dict, len_dataset: int = -1, min_num_questions: int = -1, cutoff: Optional[int] = None, seed: Optional[int] = None) -> dict:
    if len_dataset < 0 or min_num_questions < 0:
        raise ValueError("Valid (true positive) values need to be passed as kwargs for len_dataset and min_num_questions!")
    if seed is not None:
        random.seed(seed)
    # currently this seems to be a LazyRow object (see https://github.com/huggingface/datasets/blob/main/src/datasets/formatting/formatting.py)
    fields = list(table_dataset_sample.data.keys())
    # if there is a cutoff sample equal amounts of questions per table (else min_num_questions)
    max_questions_per_table = (cutoff or len_dataset*min_num_questions) // len_dataset
    # draw sample ids at random
    num_questions = len(table_dataset_sample['questions'])
    sampled_ids = random.sample(range(num_questions), k=min(max_questions_per_table, num_questions))
    # filter questions using keeping only sampled_ids
    output_dict = {field: [table_dataset_sample[field][idx]
                           for idx in sampled_ids
                           ]
                   for field in fields
                   # only filter fields that have values associated per question (same shape as 'questions' field)
                   if isinstance(table_dataset_sample[field], list) and len(table_dataset_sample[field]) == num_questions
                   }
    return output_dict


def cutoff_num_questions(dataset: datasets.Dataset, cutoff: Optional[int] = None) -> datasets.Dataset:
    min_num_questions = min([len(table_sample['questions']) for table_sample in dataset])
    return dataset.map(sample_questions,
                       fn_kwargs={
                           'len_dataset': len(dataset),
                           'min_num_questions': min_num_questions,
                           'cutoff': cutoff,
                           },
                       desc=f"Sampling a maximum of {cutoff or min_num_questions*len(dataset)} samples in total..."
                       )


def dump_mem_alloc(stop=True):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:10]:
        print(stat)
    print()
    if stop:
        tracemalloc.stop()


def main(args):
    # TODO if else for deciding whether to postproces or not (properties already created during synthesis?)
    # extract_properties_posthoc(args)
    # TODO then run tokenization

    # load tokenized data with properties
    base_filename = f"stats_{args.table_corpus}_{args.dataset_name}_tapex_tokenized"
    data_split = caching(base_filename, cache_path=args.data_dir + '/viable_tensors')
    data_split = data_split.map(extract_table_id)
    artifact = load_artifact_from_wandb('ne0ljo4e', wandb_entity='aiis-nlp')
    extended_predictions = add_column_from_artifact(data_split, artifact, {'text_prediction': 'text_predictions'})

    # compute which samples are correct
    extended_predictions = extended_predictions.map(is_prediction_correct, fn_kwargs={'comparison_fn': lambda x, y: x.strip() == y.strip()})

    # aggregate by property
    df = extended_predictions.to_pandas()
    acc_by_agg = df.groupby('aggregators')['is_correct'].sum()
    cnt_by_agg = df.groupby('aggregators')['questions'].count()
    num_row_agg_freq, num_row_agg_val = np.histogram(df['aggregation_num_rows'])
    acc_by_tab = df.groupby('table_id')['is_correct'].sum()
    cnt_by_tab = df.groupby('table_id')['questions'].count()
    question_len_freq, question_len_val = np.histogram(df['question_lengths'])
    cnt_by_answer = df.groupby('answer')['questions'].count()  # to check if there are very frequent answers like ''
    cnt_by_answer_len = df.groupby('answer_lengths')['questions'].count()
    stats = {
        'freqency_aggregators': cnt_by_agg,
        'freqency_table': cnt_by_tab,  # TODO reduce number of tables (e.g. 10-percentiles, min max mean)
        'frequency_answer': cnt_by_answer,

        #num_row_agg_freq, num_row_agg_val
    }
    print(acc_by_agg, cnt_by_agg)

# result dataset (TODO save predictions, correct)
# (TODO save aggregator as field during synthesis)
# TODO during synthesis when generating answer by sql execution save pre aggregate row count -> (post compute by answer to count aggreator for every question, but COUNT still needs to be implemented)


if __name__ == '__main__':
    # properties of interest for computing occurance/performance statistics
    #   label distribution (which answers do occur)
    #   aggregator row count (how many rows are combined)
    #   input length
    #   label length
    #   aggregator
    #   question id (to connect with predictions) --> add after tokenization / during dataloading when all joining filtering has been done
    #   model x prediction
    #   model x correct --> needs to be added after evaluation to have flexibility which target post processing to apply
    #   aggregation column
    #   aggregation column type
    #   condition operators, NOT needed because rather create one dataset split per condition type
    #       -> add unique dataset/ condition set name when joining splits in one dataset
    #   condition columns, probably too specific, leave for now
    # (optional visualizations)

    args = dargparse(DataProcessingArgs)

    # if the dataset does not have the required properties as fields yet
    # try to extract them from the question text
    # extract_properties_posthoc(args)

    main(args)
    #base_filename = f"stats_{args.table_corpus}_{args.dataset_name}_tapex_tokenized"
    #data_split = caching(base_filename, cache_path=args.data_dir + '/viable_tensors')
    #artifact = load_artifact_from_wandb('ne0ljo4e', wandb_entity='aiis-nlp')
    #extended_data = add_column_from_artifact(data_split, artifact, 'text_predictions')
    #print(artifact[0])
