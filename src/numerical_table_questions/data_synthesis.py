from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import logging.config
import re
import warnings
import weakref
from pathlib import PurePath
from typing import Tuple, List, Dict, Union, Optional, Literal

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from pandasql import sqldf
from tqdm import tqdm

from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.sql_templates import (
    SQLColumnExpression, SQLOperator, SQLOperatorTemplate,
    SQLConditionTemplate, SQLOverClauseTemplate, SQLTemplate,
    MIN, MAX, AVG, SUM, NOOP,
    find_template_variables,
)


log_file_init_path = str(PurePath(__file__).parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


# TODO replace asserts with proper Exceprions

# TODO maybe to utils
def name_id_mapping(names: List[str], both_ways: bool = False):
    name2id = {name: idx for idx, name in enumerate(names)}
    if both_ways:
        id2name = {idx: name for name, idx in name2id.items()}
        return name2id, id2name
    return name2id


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
        self._inferred_column_types = [self.infer_column_type(col)
                                       for col in self.column_names
                                       ]
        self._make_true_numeric()  # removes commas (e.g 10,000.99 -> 10000.99)
        self._preprocess_cells()   # handles trailing whitespaces and empty values
        self._col2idx, self._idx2col = name_id_mapping(self.column_names, both_ways=True)
        self._table_id = hashlib.sha256(str.encode(f'{self.size[0]}{self.size[1]}' +
                                                   ''.join(self.column_names) +
                                                   ''.join(self._inferred_column_types))
                                        ).hexdigest()

    @classmethod
    def from_state_dict(cls, state_dict):
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
    def pandas_dataframe(self):
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
            return list(self.pandas_dataframe[column_name].unique())
        else:
            return list(self.pandas_dataframe[column_name])

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
                          ) -> Literal['numeric', 'text']:
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

        if row_selection == 'first':
            sample_row_idxs = np.arange(num_test_rows)
        else:
            sample_row_idxs = np.random.choice(np.arange(len(self)),
                                               min(num_test_rows, len(self)),
                                               replace=False)
        df = self.pandas_dataframe
        sample_rows = df.iloc[sample_row_idxs, df.columns.get_loc(column_name)]

        def is_numeric(x):
            # TODO negative values
            regex = re.compile(r'(\d(,\d{3})*|\d+)?(\.\d+)?')
            return re.fullmatch(regex, x) is not None

        numeric_test_results = [is_numeric(row) for row in sample_rows]
        if all(numeric_test_results):
            return 'numeric'
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
        while len(set(column_names)) != len(column_names):
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
                if col_name_counter.get(col_name) is None:
                    col_name_counter[col_name] = 1
                    new_col_names.append(col_name)
                else:
                    col_name_counter[col_name] += 1
                    new_col_names.append(
                        col_name
                        + f"{extension_string}"
                        + f"{col_name_counter[col_name] if use_numbering else ''}"
                    )
            column_names = new_col_names
            while_counter += 1
        return column_names

    def _make_true_numeric(self):
        num_col_ids = [idx for idx, typ in enumerate(self._inferred_column_types)
                       if typ == 'numeric']
        for row in self._data_dict['rows']:
            for num_col in num_col_ids:
                row[num_col] = row[num_col].replace(',', '')

    def sample_values(self, col_name):
        raise NotImplementedError

    def _preprocess_column_names(self):
        for c, column in enumerate(self._data_dict['header']):
            # empty column names are replaced with column_ + id
            self._data_dict['header'][c] = column or f'column_{c}'

    def _preprocess_cells(self):
        for row in self.data_dict['rows']:
            for v, value in enumerate(row):
                # remove trailing whitespaces
                row[v] = value.strip()
                if value == '':
                    # all empty values are marked as such via double single quotes
                    row[v] = "''"

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
                 is_from_template: bool = False
                 ) -> None:
        self._nl_question = nl_question
        self.alternative_phrasings = []
        self._sql_query = sql_query
        self._answer = None
        self._alternative_answers = []
        self.alternative_answers = []
        self._operator = operator
        self._num_conditions = None
        self._condition_column_ids = None
        self._condition_type_id = None
        self._num_rows_aggregated_in_answer = None
        self._multi_row_answer = None
        self._table = table
        self._is_from_template = is_from_template

    def compute_answer(self) -> None:
        query_result = execute_sql(self._sql_query, self._table.pandas_dataframe)
        if len(query_result) > 1:
            warnings.warn("Query result of dataframe returned multiple rows."
                          "Queries that result in a unique answer should be "
                          "preferred."
                          )
            self._alternative_answers = [query_result.iloc[row, 0]
                                         for row in range(1, len(query_result))]
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
    # test expression format to avoid execution of malicious code
    # TODO negative values
    number_regex = r'(\d(,\d{3})*|\d+)?(\.\d+)?'  # empty string is valid number?
    full_regex = fr'"{number_regex}"|(\(?"{number_regex}"[+*/-]"{number_regex}"\)?)+'
    compiled_regex = re.compile(full_regex)
    if compiled_regex.match(expression):
        return eval(expression
                    .replace('""', '0')  # treat empty string numbers as 0
                    .replace('"', '')  # remove quotes
                    .replace(',', '')  # remove , formating (e.g 1,000,000 -> 1000000)
                    )
    else:
        warnings.warn("Provided Expression: "
                      f"'{expression[:10]+'...'+expression[-10:] if len(expression) > 20 else expression}'"
                      " is not a valid arithmetic expression and is not executed!")


class QuestionTemplate:

    def __init__(self,
                 nl_template_string,
                 sql_main_expression: SQLColumnExpression,
                 sql_allowed_operators: Tuple[SQLOperator, ...],
                 sql_conditions: Tuple[SQLConditionTemplate, ...],
                 schema,
                 template_alternatives=None
                 ):
        self._nl_template = nl_template_string
        self.template_alternatives = template_alternatives
        self.main_expression = sql_main_expression  # (SQLExpression(('{col1}',)))
        self.operators = sql_allowed_operators
        self.conditions = sql_conditions  # ([SQLCondition('{col2}', '>=', '{col3}')])
        self._schema = schema

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
                value_computation_expression = condition.condition_column.generate()
                condition_variables = {
                    'comparator': condition.comparator,
                    'column_vars': find_template_variables(value_computation_expression),
                    'value_vars': (
                        condition_val_vars
                        if len(condition_val_vars := find_template_variables(condition.value)) > 0
                        else []
                        ),
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
                                    .replace('""', '"1"')  # empty strings are also interpreted as 0 -> replace
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
                         create_alternatives=False
                         ) -> List[TableQuestion]:
        generated_questions = []
        logger.info('Creating questions from template for %i tables...', len(tables))
        for table in tqdm(tables):
            num_rows = table.size[1]
            used_datatypes = [self._schema['variables'][variable]['allowed_dtypes']
                              for variable in self._schema['variables'].keys()
                              if self._schema['variables'][variable]['type'] == 'value'
                              ]
            used_datatypes = set([dtype
                                  for var_dtypes in used_datatypes
                                  for dtype in var_dtypes])
            # TODO create Value samplers for every column
            value_sample_cols = []
            for dtype in used_datatypes:
                value_sample_cols += table.columns_by_type(dtype, names=True)
            # sampled_values = {col: table.sample_values(col) for col in value_sample_cols}
            for operator in self.operators:
                # TODO when same operator structure (e.g min, max) cache var assignments
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
                var_assignments = self.find_all_possible_assignments(sql_template,
                                                                     table
                                                                     )
                # need list of dicts not dict of lists
                compiled_sql_statements = [sql_template.format(**assignment)
                                           for assignment in var_assignments
                                           ]
                compiled_nl_questions = [self._nl_template.format(**dict(assignment,
                                                                         op=operator.text))
                                         for assignment in var_assignments
                                         ]
                if create_alternatives:
                    compiled_alternatives = [self._nl_template.format(**dict(assignment,
                                                                             op=alt))
                                             for assignment in var_assignments
                                             for alt in operator.text_alternatives
                                             ]
                    compiled_nl_questions += compiled_alternatives
                    compiled_sql_statements += (
                        len(compiled_alternatives)
                        * compiled_sql_statements
                    )
                questions = [TableQuestion(nl, table, sql, operator.sql,
                                           is_from_template=True)
                             for nl, sql in zip(compiled_nl_questions,
                                                compiled_sql_statements
                                                )
                             ]
                generated_questions.extend(questions)
        """
                # create a template for the count operator additionally to the actual aggregator in the first iteration
                # a) for statistics (field aggregation_num_rows) if do_count_augmentation is True or
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

                    seen_configurations = {}
                    for assignment in var_assignments:
                        # sampled variables for each assignment excluding the aggregation column (since count is the same for all columns)
                        # first element in assignment is always the aggregation_column by convention -> key is all but the first elements
                        configuration = ','.join([str(elem) for elem in list(assignment.values())[1:]])
                        if seen_configurations.get(configuration) is None:
                            seen_configurations[configuration] = assignment
                    # parse dict into two aligned lists of configuration hashes and variable assignments for easy processing later on
                    # initiallizes empty lists if dict is empty
                    count_configurations, count_var_assignments = zip(*seen_configurations.items()) if len(seen_configurations) > 0 else ([], [])

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
        """

        return generated_questions


class TableQuestionDataSet:

    def __init__(self,
                 name: str,
                 description: Optional[str] = None,
                 question_templates: Optional[List[QuestionTemplate]] = None,
                 tables: Optional[List[Table]] = None,
                 compute_answers=True
                 ) -> None:
        self.name = name
        self.description = description
        self._question_templates = question_templates
        self._tables = {table._table_id: table for table in (tables or [])}
        self._unanswerable_questions = []
        self._questions = self._initialize_questions(question_templates,
                                                     tables,
                                                     compute_answers=compute_answers
                                                     )
        if compute_answers:
            self._remove_unanswered_questions()
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
                artifact.compute_answer()
        else:
            raise TypeError("Argument artifact must be of type "
                            "Table, QuestionTemplate or TableQuestion!"
                            )
        self._is_answers_computed = compute_answers

    def to_huggingface(self) -> Dataset:
        """Creates a huggingface datasets.Dataset from the questions in this dataset."""
        logger.info('Grouping questions by table...')
        questions_by_table = {}
        for question in tqdm(self._questions):
            if questions_by_table.get(question._table._table_id) is None:
                questions_by_table[question._table._table_id] = {'questions': [question._nl_question],
                                                                 # TODO handle string conversion elsewhere
                                                                 'answers': [str(question._answer)]}
            else:
                questions_by_table[question._table._table_id]['questions'].append(question._nl_question)
                # TODO handle string conversion elsewhere
                questions_by_table[question._table._table_id]['answers'].append(str(question._answer))
        table = []
        questions = []
        answers = []
        logger.info('Grouping questions by table...')
        for table_id, content_dict in tqdm(questions_by_table.items()):
            table.append(self._tables[table_id].to_state_dict())
            questions.append(content_dict['questions'])
            answers.append(content_dict['answers'])
        return Dataset.from_dict({
            'table': table,
            'questions': questions,
            'answers': answers,
        })

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True, indent=4)

    def _initialize_questions(self,
                              question_templates: QuestionTemplate,
                              tables: List[Table],
                              compute_answers=True
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
        question_batches = [question_template.create_questions(tables)
                            for question_template in question_templates]
        flattened_question_list = [question
                                   for question_batch in question_batches
                                   for question in question_batch]
        if compute_answers:
            logger.info('Computing answers to table questions...')
            for question in tqdm(flattened_question_list):
                question.compute_answer()
        self._is_answers_computed = compute_answers
        return flattened_question_list

    def _remove_unanswered_questions(self) -> None:
        self._unanswerable_questions.extend([question
                                             for question in self._questions
                                             if question._answer is None]
                                            )
        self._questions = [question for question in self._questions
                           if question._answer is not None]

    def prepare_for_pickle(self):
        # removes weakrefs
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
                        ] != 'text':
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


def execute_sql(query: str, dataframe: pd.DataFrame
                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
    if query is None:
        raise ValueError("Can only compute the answer to a question if the \
                            corresponding sql_query to answer is available!"
                         )
    else:
        df = dataframe  # renaming for sqldf to find table
        try:
            query_result = sqldf(query)
        except Exception as e:
            print('query:\n', query)
            print('table:\n', dataframe.head(5))
            raise e
    return query_result


def create_table_dataset(base_dataset_name: str = 'wikitablequestions',
                         base_dataset_split: str = 'test',
                         num_tables: Optional[int] = None,
                         use_cache: bool = True,
                         cache_path: str = './data/NumTabQA/.cache'
                         ) -> Dict[str, Table]:
    cache_file_name = f'{base_dataset_name}_{base_dataset_split}_tables'
    if use_cache:
        test_tables = caching(cache_path, cache_file_name)
        if test_tables is not None:
            # restore original format by loading from state dict
            test_tables = [Table.from_state_dict(table_data) for table_data in test_tables]
    if not use_cache or test_tables is None:
        logger.info("Loading %s's first %s %s split samples",
                    base_dataset_name, str(num_tables or 'all'), base_dataset_split)
        dataset = load_dataset(base_dataset_name,
                               split=(f'{base_dataset_split}'
                                      + (f'[:{num_tables}]'
                                         if num_tables is not None
                                         else ''
                                         )
                                      )
                               )
        logger.info("Processing first %s tables of the test set...", str(num_tables or 'all'))
        # use dict comprehension to get a unique set of tables
        # by using _table_id as key duplicates are overridden
        test_tables = [Table(dataset[i]['table'],
                             source_name=base_dataset_name,
                             source_split=base_dataset_split,
                             )
                       for i in range(len(dataset))
                       ]
        for table in test_tables:
            table.prepare_for_pickle()
        save_version(test_tables, cache_path, cache_file_name)
    return test_tables


def create_basic_table_question_dataset(tables,
                                        name='wikitables_test',
                                        use_cache: bool = True,
                                        cache_path: str = './data/NumTabQA/.cache'
                                        ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_basic_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg, sum or no operation combined with a simple value lookup condition of a different column.
            Using WikiTables test set.

            """
        nl = "What is the {op} of column {col1} given that {col2} has value {val1}?"
        main_expr = SQLColumnExpression(("{col1}",))
        conditions = (SQLConditionTemplate(SQLColumnExpression(('{col2}',)), '=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, NOOP])
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric', 'text']
                         },
                'val1': {'type': 'value',
                         'allowed_dtypes': ['numeric', 'text']
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
        save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_range_table_question_dataset(tables,
                                        name='wikitables_test',
                                        use_cache: bool = True,
                                        cache_path: str = './data/NumTabQA/.cache'
                                        ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_range_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg, sum or no operation combined with a simple value lookup condition of a different column.
            Using WikiTables test set.

            """
        nl = "What is the {op} of column {col1} given that {col1} is at least {val1} and {col1} is at most {val2}?"
        main_expr = SQLColumnExpression(("{col1}",))
        conditions = (SQLConditionTemplate(SQLColumnExpression(('{col1}',)), '>=', '{val1}'),
                      SQLConditionTemplate(SQLColumnExpression(('{col1}',)), '<=', '{val2}'),
                      )
        allowed_operators = tuple([MIN, MAX, AVG, SUM])
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
        save_version(dataset, cache_path, cache_file_name)
    return dataset


def main():
    table_dataset = create_table_dataset(base_dataset_split='validation', use_cache=True)
    return create_range_table_question_dataset(table_dataset, name='wikitables_validation', use_cache=False)


if __name__ == "__main__":
    main()
