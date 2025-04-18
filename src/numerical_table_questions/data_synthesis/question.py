from __future__ import annotations

import re
import warnings
from typing import List, Optional, Union, Tuple

import datasets

from numerical_table_questions.answer_coordinates import compute_answer_coordinates
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.utils.data_utils import infer_python_type_from_str
from numerical_table_questions.utils.sql_utils import execute_sql


QUESTION_FEATURES = {'nl_question': datasets.Value('string'),
                     'alternative_phrasings': datasets.Sequence(datasets.Value('string')),
                     'sql_query': datasets.Value('string'),
                     'answer': datasets.Value('string'),  # string of numeric to keep original output number format
                     'alternative_answers': datasets.Sequence(datasets.Value('string')),
                     'answer_coordinates': datasets.Sequence(datasets.Sequence(datasets.Value('int32'))),
                     'operator': datasets.Value('string'),
                     'aggregation_column': datasets.Value('string'),
                     'condition_assignments': datasets.Sequence(datasets.Sequence(datasets.Value('string'))),
                     'num_rows_aggregated_in_answer': datasets.Value('int32'),
                     'multi_row_answer': datasets.Value('bool'),
                     'table_id': datasets.Value('string'),
                     'template_hash': datasets.Value('string'),
                     'count_hash': datasets.Value('string'),
                     }


class TableQuestion:

    def __init__(self, nl_question,
                 table: Table,
                 sql_query: Optional[str] = None,
                 operator: Optional[str] = None,
                 aggregation_column: Optional[str] = None,
                 condition_assignments: Optional[List[Tuple[str, Union[str, int, float]]]] = None,
                 _template_hash: Optional[str] = None,
                 _count_hash: Optional[str] = None,
                 ) -> None:
        self._nl_question = nl_question
        self.alternative_phrasings = []
        self._sql_query = sql_query
        self._answer = None
        self._alternative_answers = []
        self._answer_coordinates = None
        self._operator = operator
        self._agg_col = aggregation_column or self._infer_aggregation_column()
        if condition_assignments is None:  # only if explicitly None (empty list is a valid value and no need to infer)
            self._condition_assignments = self._infer_condition_assignments()
        else:
            self._condition_assignments = condition_assignments
        self._num_rows_aggregated_in_answer = None
        self._multi_row_answer = None
        self._table = table
        self._table_id = table._table_id
        self._template_hash = _template_hash
        self._count_hash = _count_hash

    def to_state_dict(self) -> dict:
        return {
            'nl_question': self._nl_question,
            'alternative_phrasings': self.alternative_phrasings,
            'sql_query': self._sql_query,
            'answer': str(self._answer or ''),
            'alternative_answers': [str(answer) for answer in self._alternative_answers],
            'answer_coordinates': self._answer_coordinates,
            'operator': self._operator,
            'aggregation_column': self._agg_col,
            # tuple of var_name + var_value (var_value can be int or str) -> cast to str for consistent feature dtype in serialization
            'condition_assignments': [(var_name, str(value))
                                      for var_name, value in self._condition_assignments
                                      ] if len(self._condition_assignments) > 0 else None,
            'num_rows_aggregated_in_answer': self._num_rows_aggregated_in_answer,
            'multi_row_answer': self._multi_row_answer,
            'table_id': self._table_id,
            'template_hash': self._template_hash,
            'count_hash': self._count_hash,
        }

    @classmethod
    def from_state_dict(cls, state_dict, table: Optional[Union[Table, List[Table], datasets.Dataset]] = None) -> TableQuestion:
        """ Creates empty instance and loads the serialized values from the state_dict
            instead of recomputing them.
        """
        instance = cls.__new__(cls)
        instance._nl_question = state_dict['nl_question']
        instance.alternative_phrasings = state_dict['alternative_phrasings']
        instance._sql_query = state_dict['sql_query']
        instance._answer = state_dict['answer']
        instance._alternative_answers = state_dict['alternative_answers']
        instance._answer_coordinates = state_dict['answer_coordinates']
        instance._operator = state_dict['operator']
        instance._agg_col = state_dict['aggregation_column']
        if state_dict['condition_assignments'] is None:
            # explicit None is only used in __init__ to tell the object to infer the condition assignments -> empty list (post-init)
            instance._condition_assignments = []
        else:
            instance._condition_assignments = [(var_name, infer_python_type_from_str(value))
                                               for var_name, value in state_dict['condition_assignments']
                                               ]
        instance._num_rows_aggregated_in_answer = state_dict['num_rows_aggregated_in_answer']
        instance._multi_row_answer = state_dict['multi_row_answer']
        instance._table = restore_table_from_id(state_dict['table_id'], table)
        instance._table_id = state_dict['table_id']
        instance._template_hash = state_dict['template_hash']
        instance._count_hash = state_dict['count_hash']
        return instance

    # TODO methods/properties for computing all attributes
    def _infer_aggregation_column(self):
        # TODO extract from sql string for cases where the question is not generated by a template
        # regex(self._sql_query)
        # TODO logger instead of print
        print("Aggregation column not explicitly provided. Inferring from SQL query (this is discouraged).")
        raise NotImplementedError("Method not implemented yet!")

    def _infer_condition_assignments(self) -> List[Tuple[str, Union[str, int, float]]]:
        # TODO extract from sql string for cases where the question is not generated by a template
        # regex(self._sql_query)
        # TODO logger instead of print
        print("Condition assignments not explicitly provided. Inferring from SQL query (this is discouraged).")
        raise NotImplementedError("Method not implemented yet!")

    @property
    def aggregation_column(self):
        return self._agg_col

    @property
    def aggregation_column_type(self):
        column_id = self._table._col2idx.get(self.aggregation_column)
        if column_id is None:
            return 'expression'
        return self._table._inferred_column_types[column_id]

    @property
    def condition_assignments(self):
        return self._condition_assignments

    @property
    def condition_columns(self):
        return [column for column, _value in self.condition_assignments]

    @property
    def condition_column_types(self):
        cond_col_ids = [self._table._col2idx.get(col) for col in self.condition_columns]
        return ['expression' if col_id is None else self._table._inferred_column_types[col_id]
                for col_id in cond_col_ids
                ]

    @property
    def num_conditions(self):
        return len(self.condition_assignments)

    @property
    def num_rows_aggregated_in_answer(self):
        return self._num_rows_aggregated_in_answer

    @property
    def multi_row_answer(self):
        return self._multi_row_answer

    @property
    def template_hash(self):
        return self._template_hash

    @property
    def is_from_template(self) -> bool:
        return self._template_hash is None

    # overwrite __eq__ and __hash__ to make set operation work as expected
    def __members(self):
        return (self._nl_question, self._sql_query, self._table_id)

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):
        return isinstance(other, TableQuestion) and self.__members() == other.__members()

    def compute_answer(self, compute_coordinates=True) -> Optional[str]:
        # TODO check if cache for same condition type and condition columns exists otherwise replace operator with count and determine _num_rows_aggregated_in_answer
        # TODO add Count question to table dataset explicitly for every condition type and condition columns hash
        query_result = execute_sql(self._sql_query, self._table.pandas_dataframe)
        if compute_coordinates:
            self._answer_coordinates = compute_answer_coordinates(self.aggregation_column, self._table.pandas_dataframe, self._sql_query)
        if query_result is not None and len(query_result) > 0:  # at least one row
            self._answer = str(query_result.iloc[0, 0])
            self._multi_row_answer = False
            # more than one row returned
            if len(query_result) > 1:
                self._multi_row_answer = True
                # do not warn because it spams the console during dataset creation;
                # maybe warn once if dataset contains at least one _multi_row_answer
                #warnings.warn("Query result of dataframe returned multiple rows."
                #              "Queries that result in a unique answer should be "
                #              "preferred."
                #              )
                self._alternative_answers = [query_result.iloc[row, 0]
                                             for row in range(1, len(query_result))
                                             ]
        else:
            self._multi_row_answer = False
            self._answer = None
        return self._answer

    def prepare_for_pickle(self):
        # remove weak reference
        self._table.prepare_for_pickle()


def compute_arithmetic_expression_str(expression: str, allowed_expression_length: int = 256):
    """ Computes the value of a valid arithmetic expression which is formatted as string.
        The numbers occuring in the expression must be enclosed in doubble quotes.
        Empty strings as numbers are interpreted as 0.
    """
    if not isinstance(expression, str):
        raise TypeError(f"Expression must be string type but is {type(expression)}!")
    if len(expression) >= allowed_expression_length:
        warnings.warn(f"Detected overly long expression {expression[:10]}...{expression[-10:]} with {len(expression)} characters! "
                      "Expression and is not executed. "
                      "Check data or increase allowed expression length."
                      )
    # test expression format to avoid execution of malicious code,
    # hence do NOT use NUMBER_REGEX constant for security reasons but rather keep it hard-coded
    number_regex = r'-?(?:(?:\.\d+)|(?:(?:\d+|(?:\d{1,3}(?:,\d{3})+))(?:\.\d+)?))'
    full_regex = fr'{number_regex}|(\(?{number_regex}[+*/-]{number_regex}\)?)+'
    compiled_regex = re.compile(full_regex)
    # column variables are wrapped in double quotes in SQL generation -> values for the columns are wrapped as well
    # -> remove double quotes before evaluation
    processed_expression = (expression
                            .replace('""', '0')  # treat empty string numbers as 0
                            .replace("''", '0')  # treat empty string numbers as 0
                            .replace('"', '')  # remove double quotes
                            .replace("'", '')  # remove single quotes
                            .replace(',', '')  # remove , formating (e.g 1,000,000 -> 1000000)
                            )
    if compiled_regex.match(processed_expression):
        return eval(processed_expression)
    else:
        warnings.warn("Provided Expression: "
                      f"'{expression[:10]+'...'+expression[-10:] if len(expression) > 20 else expression}'"
                      " is not a valid arithmetic expression and is not executed!")


def restore_table_from_id(table_id: str, table: Optional[Union[Table, List[Table], datasets.Dataset]] = None) -> Optional[Table]:
        def _search_id_in_table_list(tab_id: str, table_list: Union[List[Table], datasets.Dataset]) -> Optional[Table]:
            is_serialized = isinstance(table_list, datasets.Dataset)
            for tab in table_list:
                if is_serialized:
                    if tab['table_id'] == tab_id:
                        return Table.from_state_dict(tab)
                else:
                    if tab._table_id == tab_id:
                        return tab

        if table is None:
            warnings.warn("No table data was provided to restore Table object! Returning None. This might cause problems depending on the use-case.")
        elif isinstance(table, Table):
            if table_id != table._table_id:
                raise ValueError("Mismatch between provided Table and the table id! This would likely lead to incorrect values.")
            return table
        elif isinstance(table, (list, datasets.Dataset)):
            return _search_id_in_table_list(table_id, table)
