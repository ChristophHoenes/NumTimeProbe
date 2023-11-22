from typing import List, Type, Union

import torch

from .data_synthesis import Table, TableQuestionDataSet, execute_sql


class TableQaModel:

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def predict(self,
                str_inputs: Union[str, List[str]],
                table_inputs: Union[Table, List[Table]],
                output_tokens: bool = False
                ) -> Union[str, torch.LongTensor]:
        # TODO single vs list input detection
        if isinstance(str_inputs, list):
            str_inputs = str_inputs[:10]
        elif isinstance(str_inputs, str):
            str_inputs = [str_inputs]
        if isinstance(table_inputs, list):
            table_inputs = table_inputs[:10]
        elif isinstance(table_inputs, Table):
            table_inputs = [table_inputs]
        # TODO convert to tensors / dataframes
        tables = [table.pandas_dataframe for table in table_inputs]
        # TODO process output (parse to str)
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
        return [''.join(decoded) for sample in decoded]


def evaluate(model: Type[TableQaModel],
             dataset: TableQuestionDataSet,
             metric: str = 'exact_match_accuracy'):
    tables = [question._table for question in dataset.questions]
    predictions = model.predict([q._nl_question for q in dataset.questions],
                                tables,
                                output_tokens=metric == 'token_accuracy'
                                )
    answers = [execute_sql(sql, tables[pred].pandas_dataframe)
               for pred, sql in enumerate(predictions)]
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
