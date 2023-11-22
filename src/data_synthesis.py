from __future__ import annotations

import copy
import hashlib
import itertools
import logging
import pandas as pd
import pickle
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pandasql import sqldf
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Union, Optional, Literal

import numpy as np
from datasets import Dataset, load_dataset


logger = logging.getLogger(__name__)


# TODO replace asserts with proper Exceprions

@dataclass(frozen=True)
class SQLOperator:
    sql: str
    sql_allowed_types: Tuple[str, ...]
    text: str
    text_alternatives: Tuple[str, ...]
    sql_over_clause: bool = False
    sql_partition_cols: Tuple[str] = tuple()
    sql_order_cols: Tuple[str] = tuple()


# Pre-defined SQL window function operators
MIN = SQLOperator('min',
                  ('numeric',),
                  'minimum',
                  ('smallest value', 'lowest value', 'least value'),
                  False
                  )
MAX = SQLOperator('max',
                  ('numeric',),
                  'maximum',
                  ('largest value', 'highest value', 'biggest value', 'top value'),
                  False
                  )
AVG = SQLOperator('avg', ('numeric',), 'average', ('mean',), False)
SUM = SQLOperator('sum', ('numeric',), 'sum', ('total',), False)
NOOP = SQLOperator('', ('numeric',), 'value', tuple(), False)


class Table:

    def __init__(self, data_dict: dict,
                 name: Optional[str] = None,
                 source_name: Optional[str] = None,
                 source_split: Optional[str] = None) -> None:
        assert not (name is None and data_dict.get('name') is None), \
            "Providing a table name is mandatory!\
            If  name is not specified in data_dict it must be passed as an argument\
            explicitly."
        self._data_dict = data_dict
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
        self._col2idx = {col_name: i for i, col_name in enumerate(self.column_names)}
        self._idx2col = {idx: name for name, idx in self._col2idx.items()}
        self._table_id = hashlib.sha256(str.encode(f'{self.size[0]}{self.size[1]}' +
                                                   ''.join(self.column_names) +
                                                   ''.join(self._inferred_column_types))
                                        ).hexdigest()

    @property
    def pandas_dataframe(self):
        """This property transforms the internal structure of the table data into a
            pandas DataFrame.
        """
        return pd.DataFrame.from_dict({i: row
                                       for i, row in enumerate(self._data_dict['rows'])
                                       },
                                      orient='index',
                                      columns=self.column_names
                                      )

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

    def __len__(self):
        return self.size[1]


class TableQuestion:

    def __init__(self, nl_question,
                 table: Table,
                 sql_query: Optional[str] = None,
                 operator: Optional[str] = None,
                 is_from_template: bool = False
                 ) -> None:
        # maybe make anwer class with answer related
        # properties and a comparison function (evaluation)
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


class QuestionTemplate:
    # Think about: would it be a good approach to use answer set programming to generate
    # Templates/questions?

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
        # function to get all variables with regex {.+} out of overall sql template
        # function to get all possible assignments for variable (col names of table - 
        # crosscheck shema for type and if same value allowed simultaneously for 
        # different vars, value sampler for condition value) -> or simplify:
        # must create new template with explicit same var name to add such questions 
        self._schema = schema
        # schema variables (list of cols with each having allowed types)

        # operator (lag, lead, last, first, over partition by order by )
        # main expression (col, col_expression e.g. col - col, )
        # conditions list (col comparison value, col comparison col)
        # allow same col per category, allowed types per category
        # how to add info above (fillers)

    """
    def infer_value_columns(self) -> Dict[Union[str, Tuple[str, ...]],
                                          Union[str, Tuple[str, ...]]]:
        Infers column names that are associated with a value variable in a template.

            Args:
                template (str): The SQL template to infer the value columns from

            Returns:
                dict: Dict with keys being the value variable names and values being the 
                    associated column names 

            Todos:
                - modify docstring to hav no Args and no Returns but description to 
                    modify _schema only 

        # after var names from template are known
        value_var_column_var_map = {condition.value:condition.condition_column for condition in self.conditions}
        colum_var_idx_map = {col_var:idx for idx, col_var in enumerate(collumn_variables)}
        current_col = assignment[colum_var_idx_map[value_var_column_var_map[value]]]
        # schema[condition.value]['corresponding_column'] = condition.condition_column
        # schema['dependent_sampling'] = List[Tuple[str, ...]]
        # singles = value_vars - union(*schema['dependent_sampling'])
        """

    def find_all_possible_assignments(self,
                                      sql_template: str,
                                      table: Table
                                      ) -> List[Dict[str, str]]:
        variables = find_template_variables(sql_template)
        """
        # the following computes the cartesian product of  column_names^(len(variables))
        # -> exponential in num variables and a lot of duplicate column assignments for 
        # different variables
        column_assignments = list(itertools.product(
            *[table.column_names for i in range(len(variables))]
            )
            )
        # requires check for every assignment and each variable pair that has the same value
        # if it is desired for them to have the same column assigned
        """
        # TODO filter only type 'column' variables (maybe instead get list of values from
        # conditions and difference to all variables = col_variables)
        column_variables = [variable for variable in variables
                            if self._schema['variables'][variable]['type'] == 'column']
        # all permutations of column assignments
        column_assignments = list(itertools.permutations(table.column_names,
                                                         len(column_variables)
                                                         )
                                  )
        # check dtype constraints
        type_errors = set(
            [assignment
                for assignment in column_assignments
                for c, col in enumerate(assignment)
                if table.infer_column_type(col) not in
                self._schema['variables'][column_variables[c]]['allowed_dtypes']
             ]
        )
        column_assignments = list(set(column_assignments) - type_errors)
        # TODO pseudocode
        # I) ignore dependent values for now
        # II) samples for all columns that can ever be assigned to condition column
        # a) get ids of condition columns in assignments
        condition_vars = [var
                          for condition in self.conditions
                          for var in find_template_variables(
                              condition.condition_column.generate()
                              )
                          ]
        condition_ids = [idx
                         for idx, col_var in enumerate(column_variables)
                         if col_var in condition_vars]
        # b) set of all column names at that ids after dtype check
        condition_assignments = list(set([assignment[idx]
                                          for assignment in column_assignments
                                          for idx in condition_ids]))
        # c) determine number of value assignments hard coded or heuristic (dynamic?)
        num_value_samples = 10
        # d) iterate over results from b and call sample save in dict
        samples = {condition_col: sample_values(
                    table,
                    condition_col,
                    max_samples=num_value_samples,
                    num_draws=num_value_samples,
                    strategy=self._schema['sample_strategy'],
                    value_pool=self._schema['value_pool'],
                    shuffle=True,
                    **self._schema['interpolation_args'],
                    return_indices=False
                    ) for condition_col in condition_assignments
                   }
        # III) Iterate over assignments to zip the col_values at condition_ids' col_name
        value_assignments = [tuple(zip(*[samples[condition_col] or "''"
                                         for condition_col in [assignment[i]
                                                               for i in condition_ids]
                                         ]
                                       )) for assignment in column_assignments
                             ]
        # IV) combine with col assignments
        all_assignments = [assignment + value
                           for a, assignment in enumerate(column_assignments)
                           for value in value_assignments[a]
                           ]
        """
        # random idea not sure how to use
        condition_map = {condition.value.strip(['{', '}']):
                        condition.condition_col.strip(['{', '}'])
                        for condition in self.conditions
                        }
        # considering dependent sampling
        # 1. list of value variables
        # value_variables = [condition.value.strip(['{', '}']) for condition in self.conditions]
        value_variables = [variable for variable in variables 
                            if schema[variable]['type'] == 'value']
        # 2. get column names for every variable or dependent variable compound
        # (need extra schema argument?) that comply to dtype constraint 
        # Tuple(len=value_vars, items=Tuple(len<=col_names, items=str))
        # a) retrieve dependent value vars
        dependent_val_vars = [set([elem for elem in tup]) for tup in schema['dependent_values']]
        # b) get col names as tuples
        dependent_col_vars = 
        # c) determine and add single cols
        singles = list(set(value_variables) - set().union(*a_n_b))
        a_n_b = [tuple(elem) for elem in a_n_b]
        a_n_b.extend(singles)
        # a_n_b = new value cols -> fix naming
        value_cols = [schema[value_var]['corresponding_column'] for value_var in value_variables]  # extracted to schema from expression
        # 3. determine number of value assignments hard coded or heuristic (maybe dynamic)
        num_value_samples = 10
        # 4. sample values for each var
        # TODO think if argument num_draws is necessary or while loop
        values = [sample_values(table,
                                value_col,
                                max_samples=num_value_samples,
                                num_draws=num_value_samples,
                                strategy=schema['sample_strategy'],
                                value_pool=schema['value_pool'],
                                shuffle=True,
                                **schema['interpolation_args'],
                                return_indices=False
                                )
                for value_col in value_cols
                ]
        # 5. zip results
        value_assignments = zip(*values)
        # 6. combine with col assignments
        all_assignments = [tuple(*assignment, *value)
                        for assignment in column_assignments
                        for value in value_assignments
                        ]
        # Meta: test inter-assignment variance over different values
        """

        """
        # old implementation (incomplete/obsolete)
        if len(sample_values) > 0:
            # TODO retrieve variables' values and compute product (or smarter sampling
            #  strategy - maybe let have value generators always follow systematic pattern
            #  and then multiple shuffleings of same values can be added if shuffled 
            #  behavior is desired and then mix with zip)
            zip(sample_values.values())
            all_assignments = [tuple(*assignment, *value)
                            for assignment in column_assignments
                            for value in enumerate(sample_values.items())
                            ]
        """
        """
        value_variables = [condition.value.strip('{}')
                           for condition in self.conditions]
        """
        value_variables = [variable for variable in variables
                           if self._schema['variables'][variable]['type'] == 'value']
        ordered_variables = column_variables + value_variables
        assignments = [{variable: assingnment[v]
                        for v, variable in enumerate(ordered_variables)
                        } for assingnment in all_assignments
                       ]
        return assignments

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
                sql_template = sql_template_from_components(
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
                var_assignments = self.find_all_possible_assignments(sql_template,
                                                                     table
                                                                     )
                """
                # Variable assignment idea, replaced by function above 
                for variable in find_template_variables(sql_template):
                    if self._schema[variable]['type'] == 'column':
                        for datatype in self._schema[variable]['allowed_dtypes']:
                            for col_id in table.columns_by_type(datatype, name=False):
                                if co
                    if self._schema[variable]['type'] == 'value':
                        # TODO move value iterator initializations outside of 
                        # variable loop
                        for var_value in ValueIterator():
                            var_assignments[variable] = var_value
                """
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

        return generated_questions
        """
                for num_col_id in table.columns_by_type('numeric', name=False):
                    # TODO add group by / order by  option
                    # TODO probe column order position understanding e.g avg "of 5th
                    # column" / "of column no. 5"
                    column_name = table.column_names[num_col_id]
                    sql = "SELECT " + operator['sql'] + f"(\"{column_name}\") FROM df\n" \
                        + "WHERE true\n"
                    nl_question = f"What is the {operator['text']}" \
                        f"of column \"{column_name}\""
                    generated_questions.append(TableQuestion(nl_question + "?",
                                                             table,
                                                             sql,
                                                             operator['sql']
                                                             )
                                               )
                    # case with single text condition
                    for text_col_id in table.columns_by_type('text', name=False):
                        distinct_values = table.column_values(text_col_id, distinct=True)
                        for value in distinct_values:
                            if operator == '' and num_rows > len(distinct_values):
                                continue
                            # TODO think if extra label if condition column num_rows ~= num_distinct_values -> single-lookup all aggregates same
                            # otherwise real aggregations might be underrepresented
                            text_condition_col_name = table.col_names[text_col_id]
                            value_safe = value.replace("'", "''")
                            condition_template_sql = "\t" + f"AND \"{text_condition_col_name}\" = '{value_safe}'"
                            condition_template_text = f" where column \"{text_condition_col_name}\" has value '{value}'?"
                            numeric_intensive_dataset.add(nl_question + condition_template_text, sql + condition_template_sql, table_name, operator, condition_columns=[text_condition_col_name], condition_types=['text'], compute_answer=compute_answer)
                    # case with single numeric condition (if available)
                    # TODO find values to compare -> percentiles(may be easy if column is coded as histogram), percentiles + noise (only in one direction + or - depending on >< operator)
                    # for num_col_comp in num_cols:
                    #    # TODO decide if this constraint is desired pro avoids weird questions con can miss interesting questions like wht is the next highest number after 5, 
                    #    #if num_col == num_col_comp:
                    #    #    continue
                    #    condition_template =
                    # case with range condition on num / date (if available) special case of previous where two conditions coincide to be < and > on same column
                    # (case with condition that requires subquery computation)
                    # cases with 2 and 3 conditions (all permutations?)
        return numeric_intensive_dataset
        """


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
        self._tables = {table._table_id: table for table in tables}
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
        raise NotImplementedError

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


class SQLColumnExpression:
    """Defines the structure of an SQL column expression.

        A column expression is either a single column name or another column expression
        that combines an binary (arithmetic) operand with another column name. 
        Recursively, long expressions with multiple operands can be constructed.

        Todos:
            - add check for allowed column types (all numeric due to arithmetics?)
            - add check for same column allowed constraints across chained expressions
            - revisit if multiple arguments or always exact two 
              (see ellipsis in type annotation)
    """
    def __init__(self, arguments: Tuple[Union[str, SQLColumnExpression], ...],
                 operand: Optional[str] = None
                 ):
        self._allowed_operands = ('+', '-', '*', '/', None)
        assert operand in self._allowed_operands, \
            f"Only operands in {self._allowed_operands} allowed!"
        self.arguments = arguments
        self.operand = operand  # Maybe operand class that defines allowed types

    def generate(self):
        if self.operand is None:
            return f'"{self.arguments[0]}"'
        else:
            return (
                f'"{self.arguments[0]}"' if isinstance(self.arrguments[0], str)
                else f'({self.arrguments[0].generate()})'
                + self.operand
                + f'"{self.arguments[1]}"' if isinstance(self.arrguments[1], str)
                else f'({self.arrguments[1].generate()})'
            )


class SQLOverClauseTemplate:
    """Defines the structure of an SQL OVER clause."""

    def __init__(self, partition_columns: List[str],
                 order_columns: List[str],
                 frame_size: Tuple[Optional[int], Optional[int]] = (None, None)
                 ):
        assert len(partition_columns) > 0 or len(order_columns) > 0, \
            "Over Clause needs at least one column to perform  a partition and/or \
                order operation over!"
        self.partition_columns = partition_columns
        self.order_columns = order_columns
        self.frame_size = frame_size

    def generate(self):
        quoted_partition_columns = [f'"{col}"' for col in self.partition_columns]
        quoted_order_columns = [f'"{col}"' for col in self.order_columns]
        return (
            " OVER (" + f"\n\tPARTITION BY {','.join(quoted_partition_columns)}"
            if len(self.partition_columns) > 1 else "" +
            f"\n\tORDER BY {','.join(quoted_order_columns)}"
            if len(self.order_columns) > 1 else "" +
            f"\n\tROWS BETWEEN {self.frame_size[0] or 'UNBOUNDED'} PRECEEDING "
            f"AND {self.frame_size[1] or 'UNBOUNDED'} FOLLOWING\n)"
        )


class SQLOperatorTemplate:
    """Defines the structure of an SQL operator/aggregator (window) function).

        Todos:
            - add allowed_operators?
            - add allowed types (basic ones are all numerical but first_value can allow 
              text)
            - first_value must require over clause with ordering to be deterministic
    """

    def __init__(self,
                 operator_name: str,
                 expression: SQLColumnExpression,
                 brackets: Optional[str] = None,
                 over_clause: Optional[SQLOverClauseTemplate] = None
                 ):
        self.operator_name = operator_name
        self.brackets = brackets or "(" if self.operator_name != '' else ''
        self._allowed_brackets = {'': '', '(': ')'}
        self.expression = expression
        self.over_clause = over_clause
        self._is_integrity_valid = self.determine_integrity()

    def generate(self):
        if not self._is_integrity_valid:
            return None
        return (
            self.operator_name +
            self.brackets +
            self.expression.generate() +
            (')' if self.brackets == '(' else '') +
            (self.over_clause.generate() if self.over_clause is not None else '')
        )

    # TODO think about why not use assert instead?
    def determine_integrity(self, explain: bool = False
                            ) -> Union[bool, Tuple[bool, List[str]]]:
        """Executes integrity tests of the expression and returns results.

            If explain is set to True additional information on the reason
            for the test result of each component is provided.

            Args:
                explain (bool): Flag whether to provide explanation of
                the results or not

            Returns:
                bool: Whether the OperatorTemplate is compliant with all integrity
                      checks or not
                or tuple (if explain is True): Additional to the boolean a list of +
                                               explanations for the decision is provided
        """
        explanations = []
        if not isinstance(self.operator_name, str):
            explanations.append('operator_name must be a string')
            return False
        if not isinstance(self.expression.generate(), str) \
           or self.expression.generate() == '':
            explanations.append('expression must generate a non-empty string')
            return False
        if self.brackets not in self._allowed_brackets.keys():
            explanations.append(f'brackets must be in \
                                {list(self._allowed_brackets.keys())}'
                                )
            return False
        if self.over_clause is not None \
           and not isinstance(self.over_clause.generate(), str):
            explanations.append('over_clause must generate a string or it is invalid')
            return False
        if explain:
            return True if len(explanations) == 0 else False, explanations
        return True


class SQLConditionTemplate:
    """Defines the structure of an SQL WHERE filter condition."""

    def __init__(self,
                 condition: Union[str, SQLColumnExpression],
                 comparator: str,
                 value: Union[str, SQLColumnExpression]
                 ):
        # comparisons single(inter)/multiple(cross) column value =, <, > <=, >=
        # between range can be expressed with two conditions for now > x and < y
        # nested subquery can replace comparison value
        self._allowed_comparators = ('=', '<', '>', '<=', '>=', '!=')
        assert comparator in self._allowed_comparators, \
            "comparator must be in {self._allowed_comparators}!"
        self.comparator = comparator
        # must be of type SQLColumnExpression, str input just for ease of use -> cast
        self.condition_column = (SQLColumnExpression((condition,))
                                 if isinstance(condition, str)
                                 else condition
                                 )
        """
        # can be hard coded value but is cast to SQLColumnExpression if template string
        self.value = (SQLColumnExpression((value,))
                      if (isinstance(value, str)
                          and value.startswith('{')
                          and value.endswith('}'))
                      else value
                      )
        """
        self.value = value

    def generate(self):
        return (f"\n\tAND {self.condition_column.generate()} {self.comparator} " +
                (self.value.generate()
                 if isinstance(self.value, SQLColumnExpression)
                 else (self.value or r"''")   # two single quotes as fallback if empty
                 )
                )


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


def sql_template_from_components(operator: SQLOperatorTemplate,
                                 conditions: List[SQLConditionTemplate],
                                 table_specifier: str = 'df'
                                 ) -> str:
    """Generates final SQL template string from modular componets."""
    select_statement = "SELECT " + operator.generate() + " FROM " + table_specifier
    if len(conditions) > 0:
        select_statement += "\nWHERE true"
    for condition in conditions:
        select_statement += condition.generate()
    return select_statement


def find_template_variables(template: str) -> List[str]:
    regex_pattern = r'\{[^\{]+\}'
    return [elem.strip('{}') for elem in re.findall(regex_pattern, template)]


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


def clear_cache(cache_path: str = '../data/NumTabQA/.cache',
                prefix: Optional[str] = None,
                postfix: Optional[str] = None,
                keep_latest: bool = True
                ) -> None:
    cache_path_obj = Path(cache_path)
    cache_versions = sorted(cache_path_obj.glob((prefix or '') + '*' + (postfix or '')))
    for v, version in enumerate(cache_versions):
        if keep_latest and v == (len(cache_versions) - 1):
            break
        version.unlink()


def main():
    def create_table_dataset(base_dataset_name: str = 'wikitablequestions',
                             base_dataset_split: str = 'test',
                             num_tables: Optional[int] = None,
                             use_cache: bool = True,
                             cache_path: str = '../data/NumTabQA/.cache'
                             ) -> Dict[str, Table]:
        cache_path_obj = Path(cache_path)
        cache_file_name = f'{base_dataset_name}_{base_dataset_split}_tables.pickle'
        if cache_path_obj.exists():
            cache_versions = sorted(cache_path_obj.glob('*' + cache_file_name))
            latest_cache_version = cache_versions[-1] \
                if len(cache_versions) > 0 \
                else cache_path_obj
        else:
            cache_path_obj.mkdir(parents=True)
            latest_cache_version = cache_path_obj
        if latest_cache_version.is_file() and use_cache:
            logger.info("Loading from cache (%s)", latest_cache_version.name)
            with latest_cache_version.open('rb') as f:
                test_tables = pickle.load(f)
        else:
            logger.info("Loading %s's first %i %s split samples",
                        base_dataset_name, num_tables, base_dataset_split)
            dataset = load_dataset(base_dataset_name,
                                   split=(f'{base_dataset_split}'
                                          + (f'[:{num_tables}]'
                                             if num_tables is not None
                                             else '')
                                          )
                                   )
            logger.info("Processing first %i tables of the test set...", num_tables)
            # use dict comprehension to get a unique set of tables 
            # by using _table_id as key duplicates are overridden
            test_tables = {(tab := Table(dataset[i]['table'],
                                         source_name=base_dataset_name,
                                         source_split=base_dataset_split)
                            )._table_id: tab for i in range(len(dataset))
                           }
            save_path = cache_path_obj / (datetime.now().strftime('%y%m%d_%H%M_%S_%f_')
                                          + cache_file_name)
            logger.info("Writing list of tables to disk")
            with save_path.open('wb') as f:
                pickle.dump(test_tables, f)
        return test_tables

    def create_table_question_dataset() -> TableQuestionDataSet:
        return TableQuestionDataSet('WikiTablesBaseOperators')

    return create_table_dataset()


if __name__ == "__main__":
    main()
