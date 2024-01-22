import logging
import logging.config
import pickle
import traceback
from mock import Mock
from pathlib import PurePath
from typing import List, Type, Union

import torch
import wandb
from transformers import AutoTokenizer, TapexTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM
from lightning.pytorch.loggers import WandbLogger

import data_synthesis
from data_synthesis import Table, TableQuestionDataSet, QuestionTemplate, SQLColumnExpression, SQLOperator, SQLConditionTemplate, TableQuestion, execute_sql


log_file_init_path = str(PurePath(__file__).parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TableQaModel:

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def predict(self,
                str_inputs: Union[str, List[str]],
                table_inputs: Union[Table, List[Table]],
                output_tokens: bool = False
                ) -> Union[str, torch.LongTensor]:
        # wrap input to lists
        if isinstance(str_inputs, str):
            str_inputs = [str_inputs]
        if isinstance(table_inputs, Table):
            table_inputs = [table_inputs]
        # TODO convert to tensors / dataframes
        tables = [table.pandas_dataframe for table in table_inputs]
        encoding = self._tokenizer(table=tables,
                                   query=str_inputs,
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt")
        outputs = self._model.generate(**encoding)
        if output_tokens:
            return outputs
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def tokenize(self, input: Union[str, Table], row_format: bool = True):
        if isinstance(input, Table):
            def linearize(data_dict):
                if row_format:
                    lin_data = [col for row in data_dict['rows'] for col in row]
                else:
                    lin_data = [row[col_idx]
                                for col_idx in range(data_dict['rows'][0])
                                for row in data_dict['rows']]
                return data_dict['header'] + lin_data

            linearized_string = linearize(input._data_dict)
        else:
            linearized_string = input
        return self.tokenizer(linearized_string)


class SemanticParsingModel(TableQaModel):

    def predict(self,
                str_inputs: Union[str, List[str]],
                table_inputs: Union[Table, List[Table]],
                output_tokens: bool = False
                ) -> Union[str, torch.LongTensor]:
        # wrap input in lists
        if not isinstance(str_inputs, list):
            str_inputs = [str_inputs]
        if not isinstance(table_inputs, list):
            table_inputs = [table_inputs]

        prompt_template = ("Given the following schema:\n"
                           "{name} ({schema})\n"
                           "Write a SQL query to answer the following question:\n"
                           "{question}")
        prompt = tuple([prompt_template.format(name=table_inputs[q].table_name,
                                               schema=table_inputs[q]._data_dict['header'],
                                               question=question)
                        for q, question in enumerate(str_inputs)])
        encoding = self._tokenizer(prompt, padding=True, truncation=True,
                                   return_tensors="pt")
        # TODO remove input_ids indexing?
        outputs = self._model.generate(encoding['input_ids'], max_new_tokens=512)
        if output_tokens:
            return outputs
        decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        query_predictions = [''.join(sample) for sample in decoded]
        answers = []
        for pred, sql in enumerate(query_predictions):
            try:
                result = execute_sql(sql, table_inputs[pred].pandas_dataframe)
            except Exception as e:
                logger.warn(f"Invalid answer encounterd (setting to None)! Reason: {e}")
                result = None
            answers.append(result)
        return answers, query_predictions


def evaluate(model: Type[TableQaModel],
             dataset: TableQuestionDataSet,
             metric: str = 'exact_match_accuracy'):
    tables = [question._table for question in dataset._questions]
    predictions = model.predict([q._nl_question for q in dataset.questions],
                                tables,
                                output_tokens=metric == 'token_accuracy'
                                )
    # Semantic parsing models return the originally predicted SQL query 
    # additionally to the answer via execution
    if isinstance(predictions, tuple):
        predictions, query_predictions = predictions

    ground_truth = dataset.ground_truth
    # TODO metric dict map name tu function then call function(**kwargs)
    if metric == 'exact_match_accuracy':
        eval_results = [sample[0].strip() == sample[1].strip()
                        for sample in zip(predictions, ground_truth)]          
        metric_result = sum(eval_results) / len(eval_results)
    elif metric == 'token_accuracy':
        token_ids = [model.tokenize(gt) for gt in ground_truth]
        eval_results = [sample[0] == sample[1]
                        for sample in zip(predictions, token_ids)]
        metric_result = sum(eval_results) / len(eval_results)
    elif metric == 'fuzzy_match_accuracy':
        pass
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented!")
    return metric_result, predictions, ground_truth, eval_results


def main(model_name, dataset_version, **kwargs):
    if model_name == 'tapex':
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")
        model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wtq")
        tqa_model = TableQaModel(model, tokenizer)
    if dataset_version == 'basic_dataset':
        dataset = data_synthesis.caching('./data/NumTabQA/.cache', 'basic_dataset.pickle')
    else:
        raise FileNotFoundError(f"No saved file for dataset version '{dataset_version}' was found!")
    metric_result, predictions, ground_truth, eval_results = evaluate(tqa_model, dataset)
    """ TODO connect to wandb runs with evaluation artifacts
    evaluation_dict = {'model_name': model_name,
                       'run_details': dict(),  # TODO fill with content
                       'metric_result': metric_result,
                       'predictions': predictions,
                       'ground_truth': ground_truth,
                       'eval_results': eval_results
                       }
    pickle.dump(evaluation_dict)
    """
    print(metric_result)
    return metric_result


if __name__ == "__main__":
    args = Mock()
    args.model_name = 'tapex'
    args.dataset_version = 'basic_dataset'
    print(args.dataset_version)
    run = wandb.init(project="table-qa-debug", job_type="add-log")
    try:
        main(args.model_name, args.dataset_version)
    except:
        logger.error("Uncaught exception: %s", traceback.format_exc())
        raise SystemExit
    finally:
        artifact = wandb.Artifact("run.log", type="logfile")
        artifact.add_file("../run.log")
        wandb.log_artifact(artifact)
        
    # Test if pickled file from data_synthesis.py can be loaded here
    #with open('dummy_dataset.pkl', 'rb') as f:
    #    dummy_dataset = pickle.load(f)
    #print(dummy_dataset.name)
