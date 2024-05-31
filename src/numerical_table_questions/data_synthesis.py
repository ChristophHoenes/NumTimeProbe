from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import logging.config
import math
import re
import warnings
import weakref
from collections.abc import Iterable
from pathlib import PurePath
from typing import Tuple, List, Dict, Union, Optional, Literal

import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

from numerical_table_questions.answer_coordinates import compute_answer_coordinates
from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.sql_templates import (
    SQLColumnExpression, SQLOperator, SQLOperatorTemplate,
    SQLConditionTemplate, SQLOverClauseTemplate, SQLTemplate,
    MIN, MAX, AVG, SUM, NOOP, COUNT,
    find_template_variables,
)
from numerical_table_questions.sql_utils import execute_sql


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# NUMBER_REGEX = re.compile(r'(\d(,\d{3})*|\d+)?(\.\d+)?')  # old expression with no negative
NUMBER_REGEX = re.compile(r'-?(?:(?:\.\d+)|(?:(?:\d+|(?:\d{1,3}(?:,\d{3})+))(?:\.\d+)?))')
# TODO replace asserts with proper Exceprions


class WeakRefableDict(dict):
    pass


# TODO maybe to utils
def name_id_mapping(names: List[str], both_ways: bool = False):
    name2id = {name: idx for idx, name in enumerate(names)}
    if both_ways:
        id2name = {idx: name for name, idx in name2id.items()}
        return name2id, id2name
    return name2id


def alpha_numeric_sort(items: Iterable, indices: Optional[Iterable[int]] = None) -> Tuple[Tuple[str], Tuple[int]]:
    if indices is not None:
        if len(items) != len(indices):
            raise ValueError(f"items and indices must have same size! But they have {len(items)} and {len(indices)} items, respectively.")
        return [items[idx] for idx in indices], indices
    sorted_list = sorted([(str(item), i) for i, item in enumerate(items)])
    sorted_items, arg_sort_idxs = zip(*sorted_list)
    return sorted_items, arg_sort_idxs


class Table:

    def __init__(self, data_dict: dict,
                 name: Optional[str] = None,
                 source_name: Optional[str] = None,
                 source_split: Optional[str] = None) -> None:
        assert not (name is None and data_dict.get('name') is None), \
            "Providing a table name is mandatory!\
            If  name is not specified in data_dict it must be passed as an argument \
            explicitly."
        self._data_dict = data_dict
        # function instead of default value because uses weak reference after init
        # which requires calling (parentheses) for access
        self._dataframe = lambda: None
        self.table_name = name or self._data_dict['name']
        self._preprocess_column_names()  # fills empty column names
        self.column_names = tuple(
            self.deduplicate_column_names(self._data_dict['header'])
        )
        self._source = source_name
        self._source_split = source_split

        # data processing/cleaning and dtype inference
        self._preprocess_cells()   # handles trailing whitespaces and empty values
        # infer column types after whitespace removal
        self._inferred_column_types = [self.infer_column_type(col)
                                       for col in self.column_names
                                       ]
        self._make_true_numeric()  # removes commas (e.g 10,000.99 -> 10000.99)

        self._col2idx, self._idx2col = name_id_mapping(self.column_names, both_ways=True)
        sorted_col_names, arg_sort_col_idxs = alpha_numeric_sort(self.column_names)
        self._col_sort_idxs = arg_sort_col_idxs  # save sort idxs for later use
        # hash of table schema in terms of column names and types
        self._table_schema_id = hashlib.sha256(
            str.encode(
                ''.join(sorted_col_names) +
                ''.join(
                    alpha_numeric_sort(
                        self._inferred_column_types,
                        indices=arg_sort_col_idxs,
                        )[0]  # only hash sorted values not the sort indices
                    )
                )
            ).hexdigest()
        # sequence of sums of every numerical column (ordered alphabetically by column name) used as checksum for approximate identity
        self._num_sum_id = self._compute_num_sum_id()
        # only approximate identity for efficiency (not all values are checked)
        self._table_id = hashlib.sha256(str.encode(self._table_schema_id + self._num_sum_id + str(self.size))).hexdigest()

    @classmethod
    def from_state_dict(cls, state_dict) -> Table:
        """ Creates empty instance and loads the serialized values from the state_dict
            instead of recomputing them.
        """
        instance = cls.__new__(cls)
        instance._data_dict = state_dict['data_dict']
        instance._dataframe = lambda: None
        instance.table_name = state_dict['table_name']
        instance.column_names = state_dict['deduplicated_column_names']
        instance._source = state_dict['source_name']
        instance._source_split = state_dict['source_split']
        instance._inferred_column_types = state_dict['inferred_column_types']
        instance._col2idx, instance._idx2col = name_id_mapping(instance.column_names, both_ways=True)
        instance._table_id = state_dict['table_id']
        return instance

    @property
    def pandas_dataframe(self) -> Union[pd.DataFrame, weakref.ReferenceType[pd.DataFrame]]:
        """This property transforms the internal structure of the table data into a
            pandas DataFrame.
        """
        if self._dataframe() is None:
            df = pd.DataFrame.from_dict(
                {i: row
                 for i, row in enumerate(self._data_dict['rows'])
                 },
                orient='index',
                columns=self.column_names
            )
            self._dataframe = weakref.ref(df)
            return df
        return self._dataframe()

    @property
    def data_dict(self):
        """This property returns a deep copy of the underlying table data."""
        return copy.deepcopy(self._data_dict)

    @property
    def size(self) -> Tuple[int, int]:
        """Property size of the table in terms of number of columns and rows.

            The first item contains the number of features/columns and the second item
            the number of data poits/rows.
        """
        return len(self._data_dict['header']), len(self._data_dict['rows'])

    def _compute_num_sum_id(self):
        sums_of_sorted_numerical_columns = [self.pandas_dataframe[col].sum()
                                            for col in alpha_numeric_sort(self.column_names, self._col_sort_idxs)[0]
                                            if self._inferred_column_types[self._col2idx[col]] == 'numeric'
                                            ]
        return ';'.join([str(col_sum) for col_sum in sums_of_sorted_numerical_columns])

    def column_values(self, column_name, distinct: bool = False):
        """Determines the unique set of values occuring in a column.

            Args:
                column_name (str): The name of the column to retrieve the distinct
                values from
                distinct (bool): Whether to include duplicate values or not
                    (default:False returns columns as is without removing duplicates)

            Returns:
                list: list of (distinct) column values
        """
        if distinct:
            return list(self.pandas_dataframe.get(column_name, default=pd.Series()).unique())
        else:
            return list(self.pandas_dataframe.get(column_name, default=pd.Series()))

    def column_value_densitiy(self, column_name):
        """Calculates the ratio of distinct values and number of rows.

            Args:
                column_name (str): The name of the column to compute this metric on

            Returns:
                float: the value of the calculated metric 1.0 refers to
                    'all rows have different values' and a low value close to zero
                    indicates very sparse discrete categories
        """
        unique_values = self.column_values(column_name, distinct=True)
        return len(unique_values)/self.size[1]

    def infer_column_type(self,
                          column_name: str,
                          num_test_rows: Optional[int] = None,
                          row_selection: Literal['random', 'first'] = 'random'
                          ) -> Literal['numeric', 'text', 'alphanumeric']:
        """Assigns a data type to each column based on the string representation of its
            values.

            Uses a regular expression to match a certain pattern to infer a data type.
            Currently the following data types are distinguished:
                - numeric
                - text

            Args:
                column_name (str): Name of the column
                num_test_rows (int): Number of rows to use to infer datatype
                  if None use all rows (default)
                row_selection (str): If random samples the test rows randomly,
                  otherwise considers the first num_test_rows values

            Returns:
                str: Name of one of the predefined data types described above

            Todos:
                - extend datatypes (e.g date)
                - add description for each data type
                - implement option to frame column_name as index (Union[str, int])
        """
        # num_test_rows = None is interpreted as considering all rows
        if num_test_rows is None:
            num_test_rows = len(self)

        # determine row indices of the value samples
        if row_selection == 'first':
            sample_row_idxs = np.arange(num_test_rows)
        else:
            sample_row_idxs = np.random.choice(np.arange(len(self)),
                                               min(num_test_rows, len(self)),
                                               replace=False)
        # select sample cells from df
        df = self.pandas_dataframe
        sample_rows = df.iloc[sample_row_idxs, df.columns.get_loc(column_name)]

        # determine dtype of column based on samples
        def is_numeric(x, strict=True, number_regex=NUMBER_REGEX):
            if strict:  # true numeric (whole string must be a pure number)
                return re.fullmatch(number_regex, x) is not None
            else:
                return re.search(number_regex, x) is not None

        numeric_test_results = [is_numeric(row) for row in sample_rows]
        alphanumeric_test_results = [is_numeric(row, strict=False) for row in sample_rows]
        if all(numeric_test_results):
            return 'numeric'
        elif all(alphanumeric_test_results):
            return 'alphanumeric'
        else:
            return 'text'

    def columns_by_type(self,
                        type: str,
                        names: bool = True
                        ) -> Union[List[str], List[int]]:
        """Collects all columns of a specified type.

            Can either return a list of column names or their respective indices.

            Args:
                type (str): The datatype for which the columns should be returned

            Returns:
                list: List of strings containing all the column names that are of
                    data type 'type' or list of int with their respective indices
        """
        if names:
            return [col for col, typ in zip(self.column_names,
                                            self._inferred_column_types)
                    if typ == type
                    ]
        else:
            return [idx for idx, typ in enumerate(self._inferred_column_types)
                    if typ == type
                    ]

    def deduplicate_column_names(self,
                                 column_names: List[str],
                                 extension_string: str = "_",
                                 use_numbering=True,
                                 while_killswitch: int = 3
                                 ) -> List[str]:
        """Rename duplicate column names to get a unique set of names.

            Args:
                ...

            Todos:
                - finish Args and Returns in docstring
                - sophisticated pattern detection in repeating columns

        """
        # TODO try finding patterns of column repetitions and assign them to leftmost
        # (first column before the pattern)
        # e.g team1, score, players, team2, score, players, team3, score, ...
        # and concattenate names if they result in different pairs
        # e.g team1, team1_score, team1_players, team2, team2_score, ...
        #  else try _1, _2 ,etc.
        assert not (extension_string == "" and use_numbering is False), \
            "Either a non-empty extension_string or use_numbering=True must be used!"
        original_column_names = column_names
        while_counter = 0
        while len(set([col.lower() for col in column_names])) != len(column_names):
            if while_counter > while_killswitch:
                raise Exception(
                    f"""
                    Unusual depth of correlated/duplicated column names
                    ({original_column_names}) detected!
                    Consider using different extension_string or a higher number of
                    while_killswitch.
                    """
                )
            col_name_counter = dict()
            new_col_names = []
            for col_name in column_names:
                if col_name_counter.get(col_name.lower()) is None:
                    col_name_counter[col_name.lower()] = 1
                    new_col_names.append(col_name)
                else:
                    col_name_counter[col_name.lower()] += 1
                    new_col_names.append(
                        col_name
                        + f"{extension_string}"
                        + f"{col_name_counter[col_name.lower()] if use_numbering else ''}"
                    )
            column_names = new_col_names
            while_counter += 1
        return column_names

    def _make_true_numeric(self):
        num_col_ids = [idx for idx, typ in enumerate(self._inferred_column_types)
                       if typ == 'numeric']
        for r, row in enumerate(self._data_dict['rows']):
            for num_col in num_col_ids:
                self._data_dict['rows'][r][num_col] = row[num_col].replace(',', '')

    def sample_values(self, col_name):
        raise NotImplementedError

    def _preprocess_column_names(self):
        for c, column in enumerate(self._data_dict['header']):
            # empty column names are replaced with column_ + id
            self._data_dict['header'][c] = column or f'column_{c}'

    def _preprocess_cells(self):
        for r, row in enumerate(self._data_dict['rows']):
            for v, value in enumerate(row):
                # remove trailing whitespaces
                self._data_dict['rows'][r][v] = value.strip()
                if value == '':
                    # all empty values are marked as such via double single quotes
                    self._data_dict['rows'][r][v] = "''"

    def prepare_for_pickle(self):
        # weakref cannot be pickled, hence replace it with default value
        self._dataframe = lambda: None

    def to_state_dict(self):
        return {
            'table_id': self._table_id,
            'table_name': self.table_name,
            'source_name': self._source,
            'source_split': self._source_split,
            'data_dict': self._data_dict,
            'deduplicated_column_names': self.column_names,
            'inferred_column_types': self._inferred_column_types,
        }

    def __len__(self):
        return self.size[1]


class TableQuestion:

    def __init__(self, nl_question,
                 table: Table,
                 sql_query: Optional[str] = None,
                 operator: Optional[str] = None,
                 aggregation_column: Optional[str] = None,
                 condition_assignments: Optional[List[str]] = None,
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
        self._condition_assignments = condition_assignments or self._infer_condition_assignments()
        self._num_rows_aggregated_in_answer = None
        self._multi_row_answer = None
        self._table = table
        self._template_hash = _template_hash
        self._count_hash = _count_hash

    # TODO methods/properties for computing all attributes
    def _infer_aggregation_column(self):
        # TODO extract from sql string for cases where the question is not generated by a template
        # regex(self._sql_query)
        pass

    def _infer_condition_assignments(self):
        # TODO extract from sql string for cases where the question is not generated by a template
        # regex(self._sql_query)
        pass

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
        return (self._nl_question, self._sql_query, self._table._table_id)

    def __hash__(self):
        return hash(self.__members())

    def __eq__(self, other):
        return isinstance(other, TableQuestion) and self.__members() == other.__members()

    def compute_answer(self, compute_coordinates=True) -> None:
        # TODO check if cache for same condition type and condition columns exists otherwise replace operator with count and determine _num_rows_aggregated_in_answer
        # TODO add Count question to table dataset explicitly for every condition type and condition columns hash
        query_result = execute_sql(self._sql_query, self._table.pandas_dataframe)
        if compute_coordinates:
            self._answer_coordinates = compute_answer_coordinates(self.aggregation_column, self._table.pandas_dataframe, self._sql_query)
        if len(query_result) > 1:
            self._multi_row_answer = True
            # do not warn because it spams the console during dataset creation;
            # maybe warn once if dataset contains at least one _multi_row_answer
            #warnings.warn("Query result of dataframe returned multiple rows."
            #              "Queries that result in a unique answer should be "
            #              "preferred."
            #              )
            self._alternative_answers = [query_result.iloc[row, 0]
                                         for row in range(1, len(query_result))]
        else:
            self._multi_row_answer = False
        self._answer = query_result.iloc[0, 0] if len(query_result) > 0 else None

    def prepare_for_pickle(self):
        # remove weak reference
        self._table.prepare_for_pickle()


def compute_arithmetic_expression_str(expression: str):
    """ Computes the value of a valid arithmetic expression which is formatted as string.
        The numbers occuring in the expression must be enclosed in doubble quotes.
        Empty strings as numbers are interpreted as 0.
    """
    if not isinstance(expression, str):
        raise TypeError(f"Expression must be string type but is {type(expression)}!")
    if len(expression) >= 256:
        raise ValueError(f"Detected overly long expression with {len(expression)} characters! "
                         "Check data or increase allowed expression length.")
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


class QuestionTemplate:

    def __init__(self,
                 nl_template_string: str,
                 sql_main_expression: SQLColumnExpression,
                 sql_allowed_operators: Tuple[SQLOperator, ...],
                 sql_conditions: Tuple[SQLConditionTemplate, ...],
                 schema,
                 template_alternatives=None
                 ):
        self._nl_template = nl_template_string
        self.template_alternatives = template_alternatives
        self.main_expression = sql_main_expression  # (SQLExpression(('{col1}',)))
        self._operators, self._explicit_count_definition = self._extract_explicit_count_operator_definition(sql_allowed_operators)
        self.conditions = sql_conditions  # ([SQLCondition('{col2}', '>=', '{col3}')])
        self._schema = schema
        # approximate identity of a template that considers only the main structure of the template
        # but NOT the exact configuration options (e.g alternative phrasings, allowed operators)
        self._template_hash = hashlib.sha256(
            str.encode(self._nl_template
                       + self.main_expression.generate()
                       + ''.join([c.generate() for c in self.conditions])
                       )
            ).hexdigest()
        # TODO think which parts of the schema should be included (which parts make the resulting questions inherently different)
        # then combine with structure hash
        self._template_schema_hash=...

    def _extract_explicit_count_operator_definition(self, operators: Iterable[SQLOperator]) -> Tuple[List[SQLOperator], Optional[SQLOperator]]:
        count_definition = tuple([op for op in operators if op.sql == 'count'])

        if len(count_definition) == 0:
            count_definition = None
        elif len(count_definition) > 1:
            warnings.warn("Encountered multiple different definitions of the COUNT operator! Selecting only the first one.")
            count_definition = count_definition[0]  # return the first count definition occurance
        else:
            count_definition = count_definition[0]  # unwrap single element from tuple

        non_count_operators = tuple([op for op in operators if op.sql != 'count'])
        return non_count_operators, count_definition

    def extract_aggregation_column(self, assignment: dict):
        if self.main_expression.operand is None:
            agg_col_var_name = self.main_expression.arguments[0].strip('{}')
            return assignment.get(agg_col_var_name)
        else:
            return '_column_expression'

    def extract_condition_variables(self):
        condition_vars = []
        for condition in self.conditions:
            # add first condition argument (column expression)
            if isinstance(condition.condition_column, str):
                condition_column = condition.condition_column.strip('{}')
            elif isinstance(condition.condition_column, SQLColumnExpression):
                if condition.condition_column.operand is None:
                    condition_column = condition.condition_column.arguments[0].strip('{}')
                else:
                    condition_column = None
            else:
                raise TypeError(f"Expected condition to be of type [str, SQLColumnExpression] but got '{type(condition.condition)}'!")
            # add value variable
            if isinstance(condition.value, str):
                condition_value = condition.value.strip('{}')
            elif isinstance(condition.value, SQLColumnExpression):
                if condition.value.operand is None:
                    condition_value = condition.value.arguments[0].strip('{}')
                else:
                    condition_value = None
            condition_vars.append((condition_column, condition_value))
        return condition_vars

    def extract_condition_assignments(self, condition_vars, assignment: dict):
        return [(assignment.get(col_var_name, '_column_expression'), assignment.get(val_var_name, '_column_expression'))
                for col_var_name, val_var_name in condition_vars]

    @property
    def operators(self):
        """ Property that returns a tuple of the operators of the template.
            The count operator will be always at the first position if an explicit operator for count was specified.
        """
        if self._explicit_count_definition is not None:
            return (self._explicit_count_definition, *self._operators)
        return self._operators

    @property
    def num_conditions(self):
        return len(self.conditions)

    @property
    def template_hash(self):
        return self._template_hash

    def find_all_possible_assignments(self,
                                      sql_template: str,
                                      table: Table
                                      ) -> List[Dict[str, str]]:
        variables = find_template_variables(sql_template)
        # filter only type 'column' variables, that get assigned a column name instead of a value
        # the order in this list determines the binding of column names from column_assignments to the column variables
        # (e.g column_name at index 0 of column_assignments binds to column variable at index 0 of column_variables)
        column_variables = [variable for variable in variables
                            if self._schema['variables'][variable]['type'] == 'column']
        # mapping of column name to inferred data type
        infered_types = {name: typ for name, typ in zip(table.column_names, table._inferred_column_types)}
        # all permutations of a table's column_names for the assignment length (determind by the number of column variables to fill)
        column_assignments = list(itertools.permutations(table.column_names,
                                                         len(column_variables),
                                                         )
                                  )
        # match the names of the column variables with the assignments/bindings of the actual column names of the specific table
        column_variable_bindings = [dict(zip(column_variables, assignment)) for assignment in column_assignments]

        # check for unmet dtype constraints and filter invalid variable-column bindings
        column_variable_bindings = [
            binding
            for binding in column_variable_bindings
            if all(
                [infered_types[assigned_col] in self._schema['variables'][var_name]['allowed_dtypes']
                 for var_name, assigned_col in binding.items()
                 ]
                )
            ]

        # short circuit if no valid variable assignment was found -> no questions for this TableQuestionTemplate are generated
        if len(column_variable_bindings) == 0:
            return []

        # ignore semantic dependencies of values for now (filter empty results post hoc)

        # collect information about every condition, which is needed to decide
        # how to sample and compute the values for each value variable
        condition_info = []
        for condition in self.conditions:
            # if two SQLColumnExpressions are compared in the condition no value needs to be sampled
            if not isinstance(condition.value, SQLColumnExpression):
                value_computation_expression = condition.condition_string
                condition_variables = {
                    'comparator': condition.comparator,
                    'column_vars': find_template_variables(value_computation_expression),
                    'value_vars': find_template_variables(condition.value_string),
                    'value_computation': value_computation_expression,
                    }
            condition_info.append(condition_variables)
        # in the order of conditions assign column variables to value variables
        # to define from which columns to sample and how to compute the final value
        initiallized_value_vars = []
        value_var_cols = []
        value_var_computations = []
        value_var_comparators = []
        for info in condition_info:
            is_condition_assigned = False
            # here the order of variables does not matter since only one variable may be unassigned
            for val_var in list(set(info['value_vars'])):
                # skip all variables that were already assigned in a previous condition
                if val_var not in initiallized_value_vars:
                    if is_condition_assigned:  # if True another variable in the condition was already assigned
                        raise ValueError("Encountered more than one value variable with unassigned value! "
                                         "Please make sure all value variables (except for one) occur in a previous condition.")
                    # add assignments in order of occurance
                    initiallized_value_vars.append(val_var)
                    value_var_cols.append(set(info['column_vars']))
                    value_var_computations.append(info['value_computation'])
                    value_var_comparators.append(info['comparator'])
                    is_condition_assigned = True
        # determine how many times a column variable occurs within the conditions
        # increase factor if multiple value variables depend on the same column
        var_sampling_factors = dict()
        for col_set in value_var_cols:
            for col_var in col_set:
                if var_sampling_factors.get(col_var) is None:
                    var_sampling_factors[col_var] = 1
                else:
                    var_sampling_factors[col_var] += 1
        # the sampling factor for a column is the maximum of the factors of the variables it can be assigned to
        # hence check data type constraints to determine this
        col_sampling_factors = dict()
        for col_var, factor in var_sampling_factors.items():
            for col_name in table.column_names:
                if infered_types[col_name] in self._schema['variables'][col_var]['allowed_dtypes']:
                    if factor > col_sampling_factors.get(col_name, 0):
                        col_sampling_factors[col_name] = factor
        # how many values to sample per value variable -> TODO make configurable
        # (increases number of questions that only differ in the condition value)
        num_value_samples = 10
        # sample values for each column that can ever be assigned to a
        # condition column variable which is involved in a value computation
        samples = {
            col_name: sample_values(
                table,
                col_name,
                max_samples=num_value_samples*factor,
                num_draws=num_value_samples*factor,
                strategy=self._schema['sample_strategy'],
                value_pool=self._schema['value_pool'],
                shuffle=True,
                **self._schema['interpolation_args'],
                return_indices=False
                )
            for col_name, factor in col_sampling_factors.items()
            }
        # compute value expressions
        value_variable_assignments = []
        for binding in column_variable_bindings:  # column assignments
            # assign a sampled value to each column variable in the expression
            sample_idxs = dict()
            is_text_in_expression = False
            for _ in range(num_value_samples):  # different values for same column assignment
                computed_values = dict()
                for col_set in value_var_cols:  # every value variable that occurs (condition level)
                    value_computation_kwargs = dict()
                    for col_var in col_set:  # value components within one value variable (expression level)
                        # retrieve actual column name of current assignment
                        col_name = binding[col_var]
                        # flag to detect text in arithmetic expression
                        if infered_types[col_name] in ['text', 'alphanumeric']:
                            is_text_in_expression = True
                        # sample index (next unused sample for this column)
                        i = sample_idxs.get(col_name, 0)
                        # assign sample value to column variable
                        value_computation_kwargs[col_var] = samples[col_name][i]
                        # increment sample index to next unused value
                        sample_idxs[col_name] = i + 1
                    # postprocess numbers
                    if not is_text_in_expression:
                        # remove leading zeros
                        value_computation_kwargs = {k: v.lstrip('0') for k, v in value_computation_kwargs.items()}
                    # inject sample values into the column expression template and evaluate final value
                    for value_var, expression, col_set in zip(initiallized_value_vars, value_var_computations, value_var_cols):
                        if is_text_in_expression:
                            if '*' in expression or '/' in expression or '-' in expression:
                                raise ValueError("Only '+' operator is allowed in expression with text columns!")
                            # concatenate the assigned values as string
                            computed_value = ''.join([str(value_computation_kwargs[col_var]) for col_var in col_set])
                        else:
                            try:
                                computed_value = compute_arithmetic_expression_str(
                                    expression.format(**value_computation_kwargs)
                                    ) or "''"
                            except ZeroDivisionError:
                                warnings.warn("Encountered zero division in value computation! "
                                              "Retrying with all zeros replaced with ones. "
                                              "This might lead to unexpected values."
                                              )
                                computed_value = compute_arithmetic_expression_str(
                                    expression.format(**value_computation_kwargs)
                                    .replace('""', '"1"')  # empty strings are also interpreted as 0 -> replace with 1 for safe division
                                    .replace('0', '1')
                                    )
                        computed_values[value_var] = computed_value
                value_variable_assignments.append(computed_values)
        # ensure that a valid range is created if two values with the same
        # column expression exist with (< or <=) and (> or >=) comparators respectively
        range_variables = dict()
        for i, (expression, comparator) in enumerate(zip(value_var_computations, value_var_comparators)):
            for j, (exp, comp) in enumerate(zip(value_var_computations, value_var_comparators)):
                range_pair = (initiallized_value_vars[i], initiallized_value_vars[j])
                reverse_range_pair = (initiallized_value_vars[j], initiallized_value_vars[i])
                if range_variables.get(reverse_range_pair) is not None:
                    continue
                if expression == exp and i != j:
                    if comparator in ('<', '<=') and comp in ('>', '>='):
                        range_variables[range_pair] = dict()
                        range_variables[range_pair]['lower_bound'] = initiallized_value_vars[j]
                        range_variables[range_pair]['upper_bound'] = initiallized_value_vars[i]
                    elif comparator in ('>', '>=') and comp in ('<', '<='):
                        range_variables[range_pair] = dict()
                        range_variables[range_pair]['lower_bound'] = initiallized_value_vars[i]
                        range_variables[range_pair]['upper_bound'] = initiallized_value_vars[j]
        for assignment in value_variable_assignments:
            for var_pair, bounds in range_variables.items():
                val_1 = assignment[var_pair[0]]
                val_2 = assignment[var_pair[1]]
                if val_1 > val_2:
                    assignment[bounds['lower_bound']] = val_2
                    assignment[bounds['upper_bound']] = val_1
                else:
                    assignment[bounds['lower_bound']] = val_1
                    assignment[bounds['upper_bound']] = val_2
        # ensure same length of column assignments and value assignments
        # -> every column assignment needs to occur num_value_samples times in a row
        column_variable_bindings = [binding for binding in column_variable_bindings for i in range(num_value_samples)]
        # join column variable assignments with value variable assignments
        variable_assignments = [
            col_binding | val_assignment
            for col_binding, val_assignment in
            zip(column_variable_bindings, value_variable_assignments)
            ]
        return variable_assignments

    def create_questions(self,
                         tables: List[Table],
                         create_alternatives=False,
                         do_count_augmentation=False,
                         ) -> Tuple[List[TableQuestion], Union[List[TableQuestion], None]]:
        generated_questions = []
        generated_count_questions = [] if do_count_augmentation or self._explicit_count_definition is not None else None
        logger.info('Creating questions from template for %i tables...', len(tables))
        for table in (progress_bar := tqdm(tables)):
            progress_bar.set_description("Tables processed (all questions created)")
            num_rows = table.size[1]
            used_datatypes = [self._schema['variables'][variable]['allowed_dtypes']
                              for variable in self._schema['variables'].keys()
                              if self._schema['variables'][variable]['type'] == 'value'
                              ]
            used_datatypes = set([dtype
                                  for var_dtypes in used_datatypes
                                  for dtype in var_dtypes])
            # TODO create Value samplers for every column <- TODO check if the following3/4 lines are still nedded
            value_sample_cols = []
            for dtype in used_datatypes:
                value_sample_cols += table.columns_by_type(dtype, names=True)
            var_assignment_cache = {}
            # sampled_values = {col: table.sample_values(col) for col in value_sample_cols}

            # initiallize varible all_question_count_hashes since it is only computed if
            # do_count_augmentation or self._explicit_count_definition is not None
            all_question_count_hashes = None
            for op_idx, operator in enumerate(self._operators):
                # create Template object and generate template string for current operator (aggregator)
                sql_template_obj = SQLTemplate(
                    SQLOperatorTemplate(
                        operator.sql,
                        self.main_expression,
                        brackets=None,
                        over_clause=SQLOverClauseTemplate(
                            operator.sql_partition_cols,
                            operator.sql_order_cols,
                            operator.sql_frame_size
                        ) if operator.sql_over_clause else None
                    ),
                    conditions=self.conditions
                )
                sql_template = sql_template_obj.generate()

                # aggregator configurations with same allowed dtypes draw samples from the same distribution and can share the samples
                # TODO could be optimized even more by splitting aggregators with multiple dtypes and only sampling new aggregating col values samples once per
                # dtype (rest of schema is same within template); but maybe performance gain not worth the added complexity
                aggregator_hash = hashlib.sha256(str.encode(str(operator.sql_allowed_types))).hexdigest()
                # use cached samples for variable assignments if possible for efficiency and performance comparability between operators
                if (cached_assignments := var_assignment_cache.get(aggregator_hash)) is None:
                    var_assignments = self.find_all_possible_assignments(sql_template, table)
                    var_assignment_cache[aggregator_hash] = var_assignments
                else:
                    var_assignments = cached_assignments

                # if condition is true create a template for the count operator additionally to the actual aggregator in the first iteration
                # a) for metadata statistics (field aggregation_num_rows of the dataset) if do_count_augmentation is True or
                # b) for explicit count questions in the dataset if self._explicit_count_definition is not None
                if (do_count_augmentation or self._explicit_count_definition is not None) and op_idx == 0:
                    count_template_obj = SQLTemplate.from_template(
                        sql_template_obj,
                        operator=SQLOperatorTemplate.from_template(
                            sql_template_obj.operator,
                            operator_name='count',
                            ),
                        )
                    count_template = count_template_obj.generate()

                    all_question_count_hashes = []  # condition assignments (count_hash) of all questions
                    unique_count_configurations = {}  # map of unique count_hash to first assignment with this count_hash encountered
                    for assignment in var_assignments:
                        # sampled condition variables for each assignment (excluding the aggregation column since count is the same for all columns)
                        # first element in assignment is always the aggregation_column by convention -> count_hash key is all but the first element
                        configuration = ','.join([str(elem) for elem in list(assignment.values())[1:]])
                        all_question_count_hashes.append(configuration)
                        if unique_count_configurations.get(configuration) is None:
                            unique_count_configurations[configuration] = assignment
                    # parse dict into two aligned lists of configuration hashes and variable assignments for easy processing later on
                    if len(unique_count_configurations) == 0:  # initiallizes empty lists if dict is empty
                        count_configurations, count_var_assignments = [], []
                    else:
                        count_configurations, count_var_assignments = zip(*unique_count_configurations.items())

                    compiled_count_sql_statements = [count_template.format(**assignment)
                                                     for assignment in count_var_assignments
                                                     ]

                    # consider natural language questions and alternatives thereof only if count operator is specified explicitly
                    # (e.g. real questions should be genereted; count is not only used for statistical property determination)
                    if self._explicit_count_definition is not None:
                        compiled_count_nl_questions = [self._nl_template.format(**dict(assignment,
                                                                                       op=self._explicit_count_definition.text,
                                                                                       )
                                                                                )
                                                       for assignment in count_var_assignments
                                                       ]
                        if create_alternatives:
                            compiled_count_alternatives = [self._nl_template.format(**dict(assignment, op=alt))
                                                           for assignment in count_var_assignments
                                                           for alt in self._explicit_count_definition.text_alternatives
                                                           ]
                            compiled_count_nl_questions += compiled_count_alternatives
                            compiled_count_sql_statements += (
                                len(compiled_count_alternatives)
                                * compiled_count_sql_statements
                            )

                        count_aggregation_column_assignments = [self.extract_aggregation_column(assignment) for assignment in count_var_assignments]
                        condition_variables = self.extract_condition_variables()
                        count_condition_assignments = [self.extract_condition_assignments(condition_variables, assignment) for assignment in count_var_assignments]

                        count_questions = [
                            TableQuestion(
                                nl, table, sql, 'count',
                                aggregation_column=agg_col,
                                condition_assignments=condition_assign,
                                _template_hash=self._template_hash,
                                _count_hash=count_config,
                                )
                            for nl, sql, agg_col, condition_assign, count_config in zip(
                                compiled_count_nl_questions,
                                compiled_count_sql_statements,
                                count_aggregation_column_assignments,
                                count_condition_assignments,
                                count_configurations,
                                )
                            ]
                        generated_count_questions.extend(count_questions)
                    else:
                        count_questions = [TableQuestion('', table, sql, 'count', _template_hash=self._template_hash, _count_hash=count_config)
                                           for sql, count_config in zip(compiled_count_sql_statements, count_configurations)
                                           ]
                        generated_count_questions.extend(count_questions)

                # after potentially having initiallized the count aggregator and having computed the count_hash for all questions
                # fill the template slots for questions with the actual aggregator of the iteration
                compiled_sql_statements = [sql_template.format(**assignment)
                                           for assignment in var_assignments
                                           ]
                compiled_nl_questions = [self._nl_template.format(**dict(assignment, op=operator.text))
                                         for assignment in var_assignments
                                         ]
                if create_alternatives:
                    compiled_alternatives = [self._nl_template.format(**dict(assignment, op=alt))
                                             for assignment in var_assignments
                                             for alt in operator.text_alternatives
                                             ]
                    compiled_nl_questions += compiled_alternatives
                    # TODO answer hashing in compute answer for efficiency
                    compiled_sql_statements += (
                        len(compiled_alternatives)
                        * compiled_sql_statements
                    )

                aggregation_column_assignments = [self.extract_aggregation_column(assignment) for assignment in var_assignments]
                condition_variables = self.extract_condition_variables()
                condition_assignments = [self.extract_condition_assignments(condition_variables, assignment) for assignment in var_assignments]

                questions = [TableQuestion(nl, table, sql, operator.sql,
                                           aggregation_column=agg_col,
                                           condition_assignments=condition_cols,
                                           _template_hash=self._template_hash,
                                           _count_hash=count_config,
                                           )
                             for nl, sql, agg_col, condition_cols, count_config in zip(
                                 compiled_nl_questions,
                                 compiled_sql_statements,
                                 aggregation_column_assignments,
                                 condition_assignments,
                                 all_question_count_hashes or [None] * len(compiled_sql_statements),
                                 )
                             ]
                generated_questions.extend(questions)
        return generated_questions, generated_count_questions


class TableQuestionDataSet:

    def __init__(self,
                 name: str,
                 description: Optional[str] = None,
                 question_templates: Optional[List[QuestionTemplate]] = None,
                 tables: Optional[List[Table]] = None,
                 compute_answers=True,
                 compute_coordinates=True,
                 allow_multiple_answers=True,
                 ) -> None:
        self.name = name
        self.description = description
        self._question_templates = question_templates
        self._tables = {table._table_id: table for table in (tables or [])}
        self._unanswerable_questions = []
        self._compute_coordinates = compute_coordinates
        self._questions = self._initialize_questions(question_templates,
                                                     tables,
                                                     compute_answers=compute_answers
                                                     )
        self._questions_by_table_id = lambda: None  # callable since weakref is also callable

        # TODO define behavior when value changes
        # a) questions initiallized after value change are afected
        # b) global effect: filter all multi value answers or recompute all existing questions
        # c) no effect
        # d) make property (no change possible after init)
        self._allow_multiple_answers = allow_multiple_answers

        if compute_answers:
            self._remove_unanswered_questions()
            if not allow_multiple_answers:
                self.remove_multi_answer_questions()

        self._is_answers_computed = compute_answers

    @property
    def tables(self) -> List[Table]:
        """Property containing a shallow copy of the list of lables in the dataset."""
        return list(self._tables.values())

    @property
    def questions(self) -> List[TableQuestion]:
        """Property containing a shallow copy of the list of questions
            in the dataset.
        """
        return copy.copy(self._questions)

    @property
    def questions_by_table_id(self) -> Union[Dict[str, TableQuestion],
                                             weakref.ReferenceType[Dict[str, TableQuestion]],
                                             ]:
        if self._questions_by_table_id() is None:
            questions_by_table = WeakRefableDict()
            for question in self._questions:
                if questions_by_table.get(question._table._table_id) is None:
                    questions_by_table[question._table._table_id] = [question]
                else:
                    questions_by_table[question._table._table_id].append(question)
            self._questions_by_table_id = weakref.ref(questions_by_table)
            return questions_by_table
        return self._questions_by_table_id()

    @property
    def ground_truth(self):
        if self._is_answers_computed:
            return [question._answer for question in self._questions]
        else:
            raise AttributeError("Can only reference ground_truth when all answers are"
                                 "pre-computed!")

    def add(self,
            artifact: Union[Table, QuestionTemplate, TableQuestion],
            compute_answers=True
            ) -> None:
        """Adds a new artifact to the dataset while preserving integrity.

            Raises:
                TypeError: When type of artifact is not either Table, QuestionTemplate
                  or TableQuestion

            Todos:
                - Add support of lists
                  (entails checking that lists contain same object type)
        """
        if isinstance(artifact, Table):
            self._tables.append(artifact)
            new_questions = self._initialize_questions(
                question_templates=self._question_templates,
                tables=[artifact],
                compute_answers=compute_answers
            )
            self._questions.extend(new_questions)
        elif isinstance(artifact, QuestionTemplate):
            new_questions = self._initialize_questions(
                question_templates=[artifact],
                tables=self._tables,
                compute_answers=compute_answers
            )
            self._questions.extend(new_questions)
        elif isinstance(artifact, TableQuestion):
            self._questions.extend(artifact)
            self._tables[artifact._table._table_id] = artifact._table
            if artifact._answer is None and compute_answers:
                artifact.compute_answer(compute_coordinates=self._compute_coordinates)
        else:
            raise TypeError("Argument artifact must be of type "
                            "Table, QuestionTemplate or TableQuestion!"
                            )
        self._is_answers_computed = compute_answers

    def to_huggingface(self) -> datasets.Dataset:
        """Creates a huggingface datasets.Dataset from the questions in this dataset."""
        logger.info('Grouping questions by table...')
        # TODO refactor: use self.questions_by_table
        # TODO add column_value_densitiy feature per question
        # and maybe aggregation_value_density/diversity (count distinct/count -> additional query)
        questions_by_table = {}
        for question in (progress_bar := tqdm(self._questions)):
            progress_bar.set_description("Saving questions by table: Questions processed")
            if questions_by_table.get(question._table._table_id) is None:
                questions_by_table[question._table._table_id] = {'questions': [question._nl_question],
                                                                 'question_lengths': [len(question._nl_question)],
                                                                 # TODO handle string conversion elsewhere
                                                                 'answers': [str(question._answer)],
                                                                 'answer_lengths': [len(str(question._answer))],
                                                                 'is_multy_row_answer': [question._multi_row_answer],
                                                                 'aggregators': [question._operator],
                                                                 'aggregation_columns': [question.aggregation_column],
                                                                 'aggregation_column_types': [question.aggregation_column_type],
                                                                 'num_conditions': [question.num_conditions],
                                                                 'aggregation_num_rows': [str(question._num_rows_aggregated_in_answer)],
                                                                 }
            else:
                questions_by_table[question._table._table_id]['questions'].append(question._nl_question)
                questions_by_table[question._table._table_id]['question_lengths'].append(len(question._nl_question))
                # TODO handle string conversion elsewhere
                questions_by_table[question._table._table_id]['answers'].append(str(question._answer))
                questions_by_table[question._table._table_id]['answer_lengths'].append(len(str(question._answer)))
                questions_by_table[question._table._table_id]['is_multy_row_answer'].append(question._multi_row_answer)
                questions_by_table[question._table._table_id]['aggregators'].append(question._operator)
                questions_by_table[question._table._table_id]['aggregation_columns'].append(question.aggregation_column)
                questions_by_table[question._table._table_id]['aggregation_column_types'].append(question.aggregation_column_type)
                questions_by_table[question._table._table_id]['num_conditions'].append(question.num_conditions)
                questions_by_table[question._table._table_id]['aggregation_num_rows'].append(str(question._num_rows_aggregated_in_answer))
        table = []
        questions = []
        question_lengths = []
        answers = []
        answer_lengths = []
        is_multy_row_answer = []
        aggregators = []
        aggregation_columns = []
        aggregation_column_types = []
        num_conditions = []
        aggregation_num_rows = []
        logger.info('Grouping questions by table...')
        for table_id, content_dict in (progress_bar := tqdm(questions_by_table.items())):
            progress_bar.set_description("Saving questions by table: Tables prepared")
            table.append(self._tables[table_id].to_state_dict())
            questions.append(content_dict['questions'])
            question_lengths.append(content_dict['question_lengths'])
            answers.append(content_dict['answers'])
            answer_lengths.append(content_dict['answer_lengths'])
            is_multy_row_answer.append(content_dict['is_multy_row_answer'])
            aggregators.append(content_dict['aggregators'])
            aggregation_columns.append(content_dict['aggregation_columns'])
            aggregation_column_types.append(content_dict['aggregation_column_types'])
            num_conditions.append(content_dict['num_conditions'])
            aggregation_num_rows.append(content_dict['aggregation_num_rows'])
        return datasets.Dataset.from_dict({
            'table': table,
            'questions': questions,
            'question_lengths': question_lengths,
            'answers': answers,
            # TODO alternative answers
            'answer_lengths': answer_lengths,
            'is_multy_row_answer': is_multy_row_answer,
            'aggregators': aggregators,
            'aggregation_columns': aggregation_columns,
            'aggregation_column_types': aggregation_column_types,
            'num_conditions': num_conditions,
            'aggregation_num_rows': aggregation_num_rows,
        })

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True, indent=4)

    def _initialize_questions(self,
                              question_templates: List[QuestionTemplate],
                              tables: List[Table],
                              compute_answers=True,
                              do_count_augmentation=True,
                              ) -> List[TableQuestion]:
        """Creates the quelstions from the datasets' question templates and tables.

            The TableQuestionDataSet can also be created incrementally. If either
            self._question_templates or self._tables are empty returns an empty list.
            After initiallization the dataset can also be extended.

            Args:
                compute_answers (bool): If True also automatically computes the answer
                  for the questions (default)
                  For reduced bject initialization time set this to False.

            Returns:
                list: possibly empty list of TableQuestions
        """
        if self._question_templates is None or self._tables is None:
            return []
        # generate questions from templates and keep count operator and other operators seperate for now (more flexibility of when to
        # explicitly use count as questions vs. when to only infer pre-aggregation row count as statistical property of the question)
        question_batches, count_question_batches = zip(
            *[question_template.create_questions(tables,
                                                 do_count_augmentation=do_count_augmentation,
                                                 )
              for question_template in question_templates]
            )

        flattened_question_list = [question
                                   for question_template_batch in question_batches
                                   for question in question_template_batch]
        flattened_count_question_list = [question
                                         for question_template_batch in count_question_batches
                                         for question in question_template_batch]
        # remove duplicates
        flattened_question_list = list(set(flattened_question_list))
        flattened_count_question_list = list(set(flattened_count_question_list))

        # compute and cache the results of counting rows per condition assignment
        count_result_cache = {}
        if do_count_augmentation:
            logger.info('Computing pre-aggregation row counts of table questions...')
            for question in (progress_bar := tqdm(flattened_count_question_list)):
                progress_bar.set_description("Computing pre-aggregation row counts")
                question.compute_answer(compute_coordinates=self._compute_coordinates)
                # store count answer to be reused for similar questions (same condition)
                # although question is hashable create new hash such that all questions
                # with the same condition assignment have the same count result
                # condition_hash = hashlib.sha256(str(tuple(question.condition_assignments)).encode()).hexdigest()
                # contains all assignments explicitly while condition_assignments collapsed all multi-column expressions into one
                # TODO Fallback to second line of SQL (WHERE condition part) if no explicit _count_hash exists (see below)
                # but maybe leave out since the format of custom sql is not known (better to differentiate between custom and template versions?
                # but there is _is_from_template)
                # condition_hash = question._count_hash or hashlib.sha256(''.join(question._sql_query.split('\n')[1:])).hexdigest()
                condition_hash = question._count_hash
                count_result_cache[condition_hash] = question._answer

        # compute answers to the questions (use cached answers for questions with equivalent SQL)
        answer_result_cache = {}
        if compute_answers:
            logger.info('Computing answers to table questions...')
            for question in (progress_bar := tqdm(flattened_question_list)):
                progress_bar.set_description("Computing answers to table questions")
                if (cached_answer := answer_result_cache.get(question._sql_query)) is None:
                    question.compute_answer(compute_coordinates=self._compute_coordinates)
                    # cache answers for questions with same sql
                    answer_result_cache[question._sql_query] = question._answer
                else:
                    # do not compute same sql query twice, but use cached answer
                    question._answer = cached_answer
                self._is_answers_computed = compute_answers

            # determine which questions with count aggregator should be explicit questions in the dataset
            # rather than just used for meta data and add them
            explicit_count_questions = []
            has_template_explicit_count_question = {template.template_hash: template._explicit_count_definition is not None
                                                    for template in question_templates}
            for question in flattened_count_question_list:
                if has_template_explicit_count_question[question.template_hash]:
                    explicit_count_questions.append(question)
                    if question._answer is None:
                        question.compute_answer(compute_coordinates=self._compute_coordinates)
            flattened_question_list.extend(explicit_count_questions)

        # add row counts before aggregation as statistical property (meta data) of the TableQuestion
        if do_count_augmentation:
            for question in flattened_question_list:
                # condition_hash = hashlib.sha256(str(tuple(question.condition_assignments)).encode()).hexdigest()
                condition_hash = question._count_hash
                if count_result_cache.get(condition_hash) is None:
                    print(condition_hash)
                question._num_rows_aggregated_in_answer = count_result_cache.get(condition_hash, 'No count result')
        return flattened_question_list

    def _remove_unanswered_questions(self) -> None:
        self._unanswerable_questions.extend([question
                                             for question in self._questions
                                             if question._answer is None]
                                            )
        self._questions = [question for question in self._questions
                           if question._answer is not None]

    def remove_multi_answer_questions(self) -> None:
        self._questions = [question for question in self._questions
                           if not question._multi_row_answer]

    def remove_questions_with_lower_aggregation_count(self, threshold: int = 2, tolerance: float = 0.0) -> None:
        if tolerance > 1.0 or tolerance < 0.0:
            raise ValueError(f"tolerance must be between 0 and 1 but was {tolerance}!"
                             "It represents the allowed proportion of questions with aggregation of rows with less than threshold.")
        if tolerance == 0.0:
            self._questions = [question for question in self._questions
                               if question._operator == ''  # NOOP aggregator is kept because the whole point is to have single value
                               or (question._num_rows_aggregated_in_answer or -1) >= threshold
                               ]
        else:
            filtered_questions = []
            questions_by_table_id = self.questions_by_table_id
            for _, question_list in questions_by_table_id.items():
                num_allowed_below_threshold = math.floor(len(question_list) * tolerance)
                below_threshold_idxs = [idx
                                        for idx, question in enumerate(question_list)
                                        if (question._num_rows_aggregated_in_answer or -1) < threshold
                                        and question._operator != ''  # NOOP does not count as below threshold
                                        ]
                keep_idxs = np.random.choice(below_threshold_idxs,
                                             min(len(below_threshold_idxs), num_allowed_below_threshold),
                                             replace=False,
                                             )
                selected_table_questions = [question for idx, question in enumerate(question_list)
                                            if idx not in below_threshold_idxs or idx in keep_idxs
                                            ]
                filtered_questions.extend(selected_table_questions)
            self._questions = filtered_questions

    def prepare_for_pickle(self):
        # removes weakrefs
        self._questions_by_table_id = lambda: None
        for table in self._tables.values():
            table.prepare_for_pickle()


# TODO refactor -> break into smaller subfunctions per strategy and post processing
def sample_values(table: Table,
                  column_names: Union[str, List[str]],
                  max_samples: Optional[int] = None,
                  num_draws: int = 1,
                  strategy: str = 'random',
                  value_pool: str = 'all',
                  shuffle: bool = False,
                  interpolation_start: int = 0,
                  interpolation_end: int = 100,
                  interpolation_range_relative: bool = True,
                  return_indices=False
                  ) -> Union[List[str], List[int], List[List[str]]]:
    """Samples some representative values of a given column.

        There are different options for sampling strategies (see Arg descriptions).
        Use cases: get random subsample, get sample of distinct values, get interpolated OOD samples

        Args:
            ...

        Returns:
            list: List of column value samples formatted as strings

        Todos:
            - finish Args docstring
            - implement dtype assertions
                e.g. no interpolation possible for discrete (str) values
            - refactor strategy functions
    """
    allowed_values_strategy = ('random', 'interpolation', 'mix')
    allowed_values_value_pool = ('all', 'distinct_values', 'percentiles')
    assert strategy in allowed_values_strategy, \
        f"Unknown value {strategy} for strategy! \
        Should be one of {allowed_values_strategy}."

    assert value_pool in allowed_values_value_pool, \
        f"Unknown value {value_pool} for strategy! \
        Should be one of {allowed_values_value_pool}."

    assert isinstance(num_draws, int) and num_draws > 0, \
        "num_draws must be positive integer!"

    assert interpolation_start <= interpolation_end, \
        "Start value must be smaller or equal than end value."

    assert not (return_indices and strategy == 'interpolation'), \
        "Cannot return indices when using interpolated values!"

    assert strategy == 'random' or isinstance(column_names, str), \
        "Simultaneously sampling multiple dependent columns is only possible with \
         strategy 'random'."

    # unify format of argument to be always be a list (even with only a single item)
    # TODO think if this is
    if isinstance(column_names, str):
        column_names = [column_names]

    # default of max_samples for value_pool='percentiles' is 11 values
    # (from 0th-100th percentile in steps of 10 percent)
    # else no constraint on number of samples
    if max_samples is None:
        if value_pool == 'percentiles':
            max_samples = 11
        else:
            max_samples = float('inf')

    # check whether provided column_names exist in table
    for col in list(set(column_names)):
        assert col in table.column_names, \
            f"Encounterd column {col} which was not found in table {table.table_name} \
                (id: {table._table_id}!)"

    # dtype assertions e.g. no interpolation possible for discrete (str) values
    assert strategy == 'random' \
           or all([table._infer_column_type(column_name=col) == 'numeric'
                   for col in column_names]
                  ), "Strategies mix and interpolation only possible for numeric dtype!"

    col_values = [table.column_values(col, distinct=value_pool == 'distinct_values')
                  for col in list(set(column_names))]
    num_col_values = [len(col) for col in col_values]

    if strategy in ('interpolation', 'mix'):
        max_value = max(col_values)
        min_value = min(col_values)
        # if interpolation strategy spend entire sample budget on interpolated samples
        # if mix strategy only spend 50% (round up to next integer) on interpolation
        sample_num = max_samples if strategy == 'interpolation' \
            else max_samples // 2 + max_samples % 2
        if interpolation_range_relative:
            start_value = min_value if interpolation_start == 0 \
                else (interpolation_start/100.0) * max_value
            interpolated_values = np.linspace(start_value,
                                              (interpolation_end/100.0) * max_value,
                                              sample_num
                                              )
        else:
            interpolated_values = np.linspace(interpolation_start,
                                              interpolation_end,
                                              sample_num
                                              )
        if shuffle:
            np.random.shuffle(interpolated_values)

        if strategy == 'interpolation':
            return list(interpolated_values)
    # TODO refactor one function for random strategy and one for interpolate
    # here only call functions
    if strategy in ('random', 'mix'):
        sample_idxs = np.zeros(max_samples, dtype=int)
        filled_samples = 0
        # if mix strategy fill only remaining half of sample budget with random samples
        sample_num = max_samples if strategy == 'random' else max_samples // 2
        for draw in range(num_draws):
            if filled_samples >= max_samples:
                break
            chosen_idxs = np.random.choice(range(min(num_col_values)),
                                           min(min(num_col_values), max_samples),
                                           replace=False
                                           )
            if not shuffle and draw == 0:  # TODO maybe delete second condition draw==0 since joint sampling achieved by colum_names as lists
                chosen_idxs.sort()

            if filled_samples + len(chosen_idxs) > max_samples:
                chosen_idxs = chosen_idxs[:max_samples-filled_samples-1]

            sample_idxs[filled_samples:filled_samples+len(chosen_idxs)] = chosen_idxs
            filled_samples += len(chosen_idxs)

    if return_indices:
        return list(sample_idxs)
    else:
        random_samples = [list(np.array(column)[sample_idxs]) for column in col_values]
        if strategy == 'mix':
            return list(interpolated_values) + random_samples[0]
        else:
            for i in range(len(column_names)):
                # any type other than text stays the way it is except for empty values
                # which are represented as empty strings (double single quote character)
                if table._inferred_column_types[
                            table._col2idx[column_names[i]]
                        ] not in ['text', 'alphanumeric']:
                    for v, value in enumerate(random_samples[i]):
                        if value == '' or value is None:
                            warnings.warn('Encountered empty value. Consider checking'
                                          'sampling strategy or table data quality!'
                                          )
                            random_samples[i][v] = "''"  # single quotes represent empty
                # text values are wrapped in single quotes and any singel quote
                # character within the text is escaped with a second single quote
                else:
                    single_quote = "'"
                    sql_escaped_single_quote = "''"
                    random_samples[i] = [f"""
                                         '{sample.replace(single_quote,
                                                          sql_escaped_single_quote
                                                          )
                                           }'
                                         """
                                         for sample in random_samples[i]]
            return random_samples[0] if len(column_names) == 1 else random_samples


def load_table_dataset(table_corpus: str = 'wikitables',
                       split: Optional[str] = None,
                       cache_path: str = './data/NumTabQA/.cache',
                       ) -> Optional[List[Table]]:
    cache_file_name = f"{table_corpus}_{split or 'all'}_tables"
    tables = caching(cache_file_name, cache_path=cache_path)
    if tables is not None:
        # restore original format by loading from state dict
        tables = [Table.from_state_dict(table_data) for table_data in tables]
    return tables


def create_table_dataset(base_dataset_name: str = 'wikitablequestions',
                         base_dataset_split: str = 'test',
                         num_tables: Optional[int] = None,
                         use_cache: bool = True,
                         cache_path: str = './data/NumTabQA/.cache',
                         save: bool = True,
                         ) -> Dict[str, Table]:
    cache_file_name = f'{base_dataset_name}_{base_dataset_split}_tables'
    if use_cache:
        tables = load_table_dataset(base_dataset_name, base_dataset_split, cache_path)
    if not use_cache or tables is None:
        logger.info("Loading %s's first %s %s split samples",
                    base_dataset_name, str(num_tables or 'all'), base_dataset_split)
        dataset_slice = '[:{num_tables}]' if num_tables is not None else ''
        dataset = datasets.load_dataset(
            base_dataset_name,
            split=f'{base_dataset_split}{dataset_slice}',
            )
        logger.info("Processing first %s tables of the test set...", str(num_tables or 'all'))
        # generate table object for every question in the source data split and use dict
        # to get a unique set of tables, by using _table_id as key near duplicates are overridden
        # Caution: _table_id does not test exact table equality of all values but only a proxy
        unique_tables = {}
        for i in range(len(dataset)):
            table = Table(dataset[i]['table'],
                          source_name=base_dataset_name,
                          source_split=base_dataset_split,
                          )
            unique_tables[table._table_id] = table
        tables = list(unique_tables.values())

        if save:
            for table in tables:
                table.prepare_for_pickle()
            save_version(tables, cache_path, cache_file_name)
    return tables


def create_basic_table_question_dataset(tables,
                                        name='wikitables_test',
                                        use_cache: bool = True,
                                        cache_path: str = './data/NumTabQA/.cache',
                                        save=True,
                                        ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_basic_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum combined with a simple value lookup condition of a different column.
            Using WikiTables test set.

            """
        nl = "What is the {op} of column {col1} given that {col2} has value {val1}?"
        main_expr = SQLColumnExpression(("{col1}",))
        conditions = (SQLConditionTemplate('{col2}', '=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, COUNT])  # not NOOP because it would be simple lookup without numerical skill
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                         },
                'val1': {'type': 'value',
                         'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                         }
            },
            'sample_strategy': 'random',
            'value_pool': 'distinct_values',
            'interpolation_args': dict()
        }
        basic_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_basic',
                                       description=base_description,
                                       question_templates=[basic_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_range_table_question_dataset(tables,
                                        name='wikitables_test',
                                        use_cache: bool = True,
                                        cache_path: str = './data/NumTabQA/.cache',
                                        save=True,
                                        ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_range_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum combined within a range of the same column (subset).
            Using WikiTables test set.

            """
        nl = "What is the {op} of column {col1} given that {col1} is at least {val1} and {col1} is at most {val2}?"
        main_expr = SQLColumnExpression(("{col1}",))
        conditions = (SQLConditionTemplate('{col1}', '>=', '{val1}'),
                      SQLConditionTemplate('{col1}', '<=', '{val2}'),
                      )
        allowed_operators = tuple([MIN, MAX, AVG, SUM, COUNT])
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'val1': {'type': 'value',
                         'allowed_dtypes': ['numeric']
                         },
                'val2': {'type': 'value',
                         'allowed_dtypes': ['numeric']
                         },
            },
            'sample_strategy': 'random',
            'value_pool': 'distinct_values',
            'interpolation_args': dict()
        }
        range_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_range',
                                       description=base_description,
                                       question_templates=[range_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_basic_postprocessed_versions(cache_path: str = './data/NumTabQA/.cache'):
    # train only filter multi answer
    table_dataset = create_table_dataset(base_dataset_split='train', use_cache=True)
    base_name = 'count_wikitables_train'
    dataset = create_basic_table_question_dataset(table_dataset, name=base_name, use_cache=False)
    dataset.remove_multi_answer_questions()
    #save_version(dataset, cache_path, base_name + '_filtered_multi_answer')
    dataset.remove_questions_with_lower_aggregation_count(tolerance=0.2)
    save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_20')
    dataset.remove_questions_with_lower_aggregation_count(tolerance=0.0)
    save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_0')

    # validation filter multi answer + 20%/0% tolerance agg_count 1
    #table_dataset = create_table_dataset(base_dataset_split='validation', use_cache=True)
    #base_name = 'count_wikitables_validation'
    #dataset = create_basic_table_question_dataset(table_dataset, name=base_name, use_cache=False)
    #dataset.remove_multi_answer_questions()
    #dataset.remove_questions_with_lower_aggregation_count(tolerance=0.2)
    #save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_20')
    #dataset.remove_questions_with_lower_aggregation_count(tolerance=0.0)
    #save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_0')

    # test filter multi answer + 20%/0% tolerance agg_count 1
    #table_dataset = create_table_dataset(base_dataset_split='test', use_cache=False)
    #base_name = 'count_wikitables_test'
    #dataset = create_basic_table_question_dataset(table_dataset, name=base_name, use_cache=False)
    #dataset.remove_multi_answer_questions()
    #dataset.remove_questions_with_lower_aggregation_count(tolerance=0.2)
    #save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_20')
    #dataset.remove_questions_with_lower_aggregation_count(tolerance=0.0)
    #save_version(dataset, cache_path, base_name + '_filtered_multi_answer_filter_agg_count_0')


def remove_duplicate_qa_pairs(data_sample):
    """ Removes duplicate QA pairs within the table batch.
        This should only be necessary for old datasets.
        In newer versions duplicats should already have been removed during dataset synthesis.
    """
    unique_questions, unique_answers = list(zip(
        *set(zip(data_sample['questions'], data_sample['answers']))
        ))
    return {'questions': unique_questions, 'answers': unique_answers}


def main():
    #table_dataset = create_table_dataset(base_dataset_split='validation', use_cache=False)
    #return create_basic_table_question_dataset(table_dataset, name='count_wikitables_validation', use_cache=False)
    #create_basic_postprocessed_versions()
    table_dataset = load_table_dataset(table_corpus='gittables_subset_10', split='train', cache_path='/home/mamba/.cache')
    # run column_name deduplication (since code changed since table dataset creation)
    for table in table_dataset:
        table.column_names = tuple(table.deduplicate_column_names(table.column_names))
        table._col2idx, table._idx2col = name_id_mapping(table.column_names, both_ways=True)
    return create_basic_table_question_dataset(table_dataset, name='gittables_subset_10_train', use_cache=False, cache_path='/home/mamba/.cache')


if __name__ == "__main__":
    main()
