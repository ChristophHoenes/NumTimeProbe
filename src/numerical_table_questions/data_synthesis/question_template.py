from __future__ import annotations

import hashlib
import itertools
import logging
import math
import random
import warnings
from collections.abc import Iterable
from pathlib import PurePath
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from numerical_table_questions.data_synthesis.question import TableQuestion, compute_arithmetic_expression_str
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.sql_templates import (
    SQLColumnExpression, SQLConditionTemplate, SQLOperator, SQLOperatorTemplate, SQLOverClauseTemplate, SQLTemplate,
    find_template_variables, get_operator_by_name
)


log_file_init_path = str(PurePath(__file__).parent.parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def sample_assignments(allowed_columns_map: Dict[str, List[str]],  # column variables to allowed columns mapping
                       num_assignments: int,
                       retries: int = 5,
                       ):
    column_variable_bindings = []
    for _ in range(num_assignments):
        for r in range(retries):
            assignment_dict = {}
            already_assigned_cols = []
            for col_var, allowed_cols in allowed_columns_map.items():
                available_columns = list(set(allowed_cols) - set(already_assigned_cols))
                if len(available_columns) == 0:
                    warnings.warn(f"Could not find a valid column assignment for column variable '{col_var}' from columns {allowed_cols}! Retry {r+1} of {retries}.")
                    break  # abort insatisfiable assignment
                drawn_col = random.choice(available_columns)
                assignment_dict[col_var] = drawn_col
                already_assigned_cols.append(drawn_col)
            if len(assignment_dict) == len(allowed_columns_map.keys()):
                # only add assignment if it is not already present in the list (else retry)
                if not any([assignment_dict == assignment for assignment in column_variable_bindings]):
                    column_variable_bindings.append(assignment_dict)
                    break  # abort retries on valid assignment
    return column_variable_bindings


class QuestionTemplate:

    def __init__(self,
                 nl_template_string: str,
                 sql_main_expression: SQLColumnExpression,
                 sql_allowed_operators: Tuple[SQLOperator, ...],
                 sql_conditions: Tuple[SQLConditionTemplate, ...],
                 schema: dict,
                 template_alternatives: Optional[List[str]] = None
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

    def to_state_dict(self) -> dict:
        return {
            'nl_template_string': self._nl_template,
            'template_alternatives': self.template_alternatives,
            'sql_main_expression': self.main_expression.to_state_dict(),
            'sql_allowed_operators': [operator.sql for operator in self._operators],
            'sql_conditions': [{'condition': condition.condition_column,
                                'comparator': condition.comparator,
                                'value': condition.value
                                }
                               for condition in self.conditions
                               ],
            'schema': self._schema,
            'template_hash': self._template_hash,
        }

    @classmethod
    def from_state_dict(cls, state_dict) -> QuestionTemplate:
        """ Creates empty instance and loads the serialized values from the state_dict
            instead of recomputing them.
        """
        instance = cls.__new__(cls)
        instance._nl_template = state_dict['nl_template_string']
        instance.template_alternatives = state_dict['template_alternatives']
        instance.main_expression = SQLColumnExpression.from_state_dict(state_dict['sql_main_expression'])
        # analogously to __init__ separate count operator and other operators
        sql_allowed_operators = tuple([get_operator_by_name(op) for op in state_dict['sql_allowed_operators']])
        instance._operators, instance._explicit_count_definition = instance._extract_explicit_count_operator_definition(sql_allowed_operators)
        instance.conditions = tuple([SQLConditionTemplate(condition=condition['condition'],
                                                          comparator=condition['comparator'],
                                                          value=condition['value'],
                                                          )
                                     for condition in state_dict['sql_conditions']
                                     ]
                                    )
        instance._schema = state_dict['schema']
        # postprocess schema variables
        if instance._schema is not None and (schema_variables := instance._schema.get('variables')) is not None:
            marked_for_deletion = []
            for var_name, val in schema_variables.items():
                # delete empty variable keys that were added by datasets.Dataset in order to have consistent nested features
                # otherwise can lead to problems in question generation (filling non existent variables with values)
                if val is None:
                    marked_for_deletion.append(var_name)
            for var_name in marked_for_deletion:
                del schema_variables[var_name]
        instance._template_hash = state_dict['template_hash']
        return instance

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

    def extract_aggregation_column(self, assignment: dict) -> str:
        if self.main_expression.operand is None:
            # arguments should always be tuple and if operand is None should be a tuple with one string element
            agg_col_var_name = self.main_expression.arguments[0].strip('{}')
            return assignment.get(agg_col_var_name)
        else:
            return '_column_expression'

    def extract_condition_variables(self) -> List[Tuple[str, str]]:
        condition_vars = []
        for condition in self.conditions:
            # add first condition argument (column expression)
            if isinstance(condition.condition_column, str):
                condition_column = condition.condition_column.strip('{}')
            elif isinstance(condition.condition_column, SQLColumnExpression):
                if condition.condition_column.operand is None:
                    condition_column = condition.condition_column.arguments[0].strip('{}')
                else:
                    condition_column = '_column_expression'
            else:
                raise TypeError(f"Expected condition to be of type [str, SQLColumnExpression] but got '{type(condition.condition)}'!")
            # add value variable
            if isinstance(condition.value, str):
                condition_value = condition.value.strip('{}')
            elif isinstance(condition.value, SQLColumnExpression):
                if condition.value.operand is None:
                    condition_value = condition.value.arguments[0].strip('{}')
                else:
                    condition_value = '_expression_value'
            condition_vars.append((condition_column, condition_value))
        return condition_vars

    def extract_condition_assignments(self, condition_vars: List[Tuple[str, str]], assignment: dict) -> List[Tuple[str, str]]:
        return [(assignment.get(col_var_name, '_column_expression'), assignment.get(val_var_name, '_expression_value'))
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

    def get_sql_template(self,
                         operator: SQLOperatorTemplate = None,
                         brackets: Optional[str] = None,
                         ) -> SQLTemplate:
        SQLTemplate(
            SQLOperatorTemplate(
                operator.sql if operator else '{op}',
                self.main_expression,
                brackets=brackets,
                over_clause=SQLOverClauseTemplate(
                    operator.sql_partition_cols,
                    operator.sql_order_cols,
                    operator.sql_frame_size
                ) if operator and operator.sql_over_clause else None
            ),
            conditions=self.conditions
        )

    def find_all_possible_assignments(self,
                                      sql_template: str,
                                      table: Table,
                                      max_num_value_samples: int = 10,
                                      max_value_length: Optional[int] = None,
                                      max_num_questions: Optional[int] = None,  # limit questions on same table if many columns
                                      retries: int = 5,  # number of retries to find max_num_questions without generating all column permutations
                                      # oversample the assignments (diversity_factor*max_num_questions) to increase the chance of sampling max_num_questions from different assignments
                                      diversity_factor: int = 2,
                                      ) -> List[Dict[str, str]]:
        variables = find_template_variables(sql_template)
        # filter only type 'column' variables, that get assigned a column name instead of a value
        # the order in this list determines the binding of column names from column_assignments to the column variables
        # (e.g column_name at index 0 of column_assignments binds to column variable at index 0 of column_variables)
        column_variables = [variable for variable in variables
                            if self._schema['variables'][variable]['type'] == 'column']
        # mapping of column name to inferred data type
        infered_types = {name: typ for name, typ in zip(table.column_names, table._inferred_column_types)}
        if max_value_length is not None:
            allowed_cols_by_col_var = {col_var: [col_name for col_name, col_typ in infered_types.items() if col_typ in self._schema['variables'][col_var]['allowed_dtypes']] for col_var in column_variables}
            column_variable_bindings = sample_assignments(allowed_cols_by_col_var, diversity_factor*max_num_questions, retries)
        else:
            # all permutations of a table's column_names for the assignment length (determind by the number of column variables to fill)
            # TODO make sure that the permutations do not become to big e.g compute n! / (n-r)! (n=column_names, r=len(col_vars))
            # ...and if higher than threshold reduce n until threshold is met (sample (preferably numerical?) columns for subset of n)
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
        estimated_question_number = len(column_variable_bindings) * max_num_value_samples
        if max_num_questions is not None and estimated_question_number > max_num_questions:
            num_value_samples = max(1, math.floor(max_num_value_samples * (max_num_questions/estimated_question_number)))
        else:
            num_value_samples = max_num_value_samples
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
        #print('pre-filter', len(column_variable_bindings))
        #print('max_value_length', max_value_length)
        #print('value_lengths', [len(str(val)) for val_assignment in value_variable_assignments for val in val_assignment.values()])
        variable_assignments = [
            col_binding | val_assignment
            for col_binding, val_assignment in
            zip(column_variable_bindings, value_variable_assignments)
            # filter assignments with values that exceed max_value_length characters (if max_value_length is None the filter is inactive)
            if max_value_length is None or all([len(str(val)) <= max_value_length for val in val_assignment.values()])
            ]
        #print('post-filter', len(variable_assignments))
        # reduce number of assignments to max_num_questions
        question_number = len(variable_assignments)
        if max_num_questions is not None and question_number > max_num_questions:
            # randomly sample max_num_questions assignments
            variable_assignments = random.sample(variable_assignments, max_num_questions)
        #print('post-trunc', len(variable_assignments))
        return variable_assignments

    def create_questions(self,
                         tables: List[Table],
                         create_alternatives=False,
                         do_count_augmentation=False,
                         max_num_value_samples: int = 10,
                         max_value_length: Optional[int] = None,
                         max_questions_per_table: Optional[int] = None,
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
                # TODO use self.get_sql_template()
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
                    var_assignments = self.find_all_possible_assignments(sql_template,
                                                                         table,
                                                                         max_num_value_samples=max_num_value_samples,
                                                                         max_value_length=max_value_length,
                                                                         max_num_questions=max_questions_per_table,
                                                                         )
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
                    random_samples[i] = [f"'{sample.replace(single_quote, sql_escaped_single_quote)}'"
                                         for sample in random_samples[i]
                                         ]
            return random_samples[0] if len(column_names) == 1 else random_samples
