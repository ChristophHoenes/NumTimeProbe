import copy
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePath

import datasets
import numpy as np
from dargparser import dargparse
from typing import Dict, List, Optional, Tuple, Union

from numerical_table_questions.arguments import DataProcessingArgs
from numerical_table_questions.utils.data_caching import save_version, caching
from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet, ensure_table_dataset_on_disk
from numerical_table_questions.data_synthesis.question_template import QuestionTemplate
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.table_creation import create_table_dataset, load_table_dataset
from numerical_table_questions.utils.data_utils import get_cache_path
from numerical_table_questions.data_synthesis.sql_templates import (
    SQLColumnExpression, SQLConditionTemplate, SQLOperator,
    MIN, MAX, SUM, AVG, COUNT, NOOP, find_template_variables,
)


BASIC_TEMPLATE = QuestionTemplate(
    nl_template_string="What is the {op} of column {col1} given that {col2} has value {val1}?",
    sql_main_expression=SQLColumnExpression(("{col1}",)),
    sql_allowed_operators=tuple([MIN, MAX, AVG, SUM, COUNT]),  # not NOOP because it would be simple lookup without numerical skill
    sql_conditions=(SQLConditionTemplate('{col2}', '=', '{val1}'),),
    schema={
        'variables': {
            'col1': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col2': {
                'type': 'column',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                },
            'val1': {
                'type': 'value',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                }
            },
        'sample_strategy': 'random',
        'value_pool': 'distinct_values',
        'interpolation_args': dict(),
        },
    template_alternatives=None
)

# TODO post-hoc add-one/distractor dataset from basic Template by sampling random distractor and updating answer according to operator (and pre-aggregation count)

DIFF_TEMPLATE = QuestionTemplate(
    nl_template_string="What is the {op} of the difference between column {col1} and {col2} given that {col3} has value {val1}?",
    sql_main_expression=SQLColumnExpression(("{col1}", "{col2}"), operand='-'),
    sql_allowed_operators=tuple([MIN, MAX, AVG, SUM, COUNT, NOOP]),
    sql_conditions=(SQLConditionTemplate('{col3}', '=', '{val1}'),),
    schema={
        'variables': {
            'col1': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col2': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col3': {
                'type': 'column',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                },
            'val1': {
                'type': 'value',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                }
            },
        'sample_strategy': 'random',
        'value_pool': 'distinct_values',
        'interpolation_args': dict()
        },
    template_alternatives=None
)


RATIO_NO_FILTER_TEMPLATE = QuestionTemplate(
    nl_template_string="What is the {op} of the ratio between column {col1} and column {col2}?",
    sql_main_expression=SQLColumnExpression(("{col1}", "{col2}"), operand='/'),
    sql_allowed_operators=tuple([MIN, MAX, AVG, SUM, COUNT, NOOP]),
    sql_conditions=tuple(),
    schema={
        'variables': {
            'col1': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col2': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            },
        'sample_strategy': 'random',
        'value_pool': 'distinct_values',
        'interpolation_args': dict()
        },
    template_alternatives=None
)

EXPRESSION_TEMPLATE = QuestionTemplate(
    nl_template_string="What is the {op} of the expression  between column {col1} and column {col2}?",
    sql_main_expression=SQLColumnExpression(
        (SQLColumnExpression(
            (SQLColumnExpression(("{col1}", "{col2}"), operand='*'),
             SQLColumnExpression(("{col1}", "{col2}"), operand='+'),
             ),
            operand='-',
            ),
         SQLColumnExpression(
             (SQLColumnExpression(("{col1}", "{col1}"), operand='*'),
              SQLColumnExpression(
                  (SQLColumnExpression(("{col2}", "{col2}"), operand='*'),
                   "{col2}"
                   ),
                  operand='-',
                  ),
              ),
             operand='+',
             )
         ),
        operand='/',
        ),
    # string representation TODO funtion for automated parsing from string
    #'(("{col1}" * "{col2}") - ("{col1}" + "{col2}")) / ("{col1}" * "{col1}") + (("{col2}" * "{col2}") - "{col2}")'
    sql_allowed_operators=tuple([MIN, MAX, AVG, SUM, COUNT, NOOP]),  # TODO maybe not COUNT because to boring? Same as in other datasets
    sql_conditions=(SQLConditionTemplate('{col3}', '=', '{val1}'),),
    schema={
        'variables': {
            'col1': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col2': {
                'type': 'column',
                'allowed_dtypes': ['numeric']
                },
            'col3': {
                'type': 'column',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                },
            'val1': {
                'type': 'value',
                'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                }
            },
        'sample_strategy': 'random',
        'value_pool': 'distinct_values',
        'interpolation_args': dict()
        },
    template_alternatives=None
)


def get_template_by_name(template_name: Union[str, List[str]]) -> Optional[Union[QuestionTemplate, List[QuestionTemplate]]]:
    is_single_template = False
    if isinstance(template_name, str):
        is_single_template = True
        template_name = [template_name]
    template_list = []
    for name in template_name:
        match name.lower():
            case 'basic': template_list.append(BASIC_TEMPLATE)
            case 'diff' | 'difference': template_list.append(DIFF_TEMPLATE)
            case 'ratio_no_filter': template_list.append(RATIO_NO_FILTER_TEMPLATE)
            case 'expression' | 'expr': template_list.append(EXPRESSION_TEMPLATE)
            case 'standard_templates': template_list.extend(get_standard_templates())
            case _:
                warnings.warn(f"No template(s) could be found for the name {name}! Trying to find template dataset at this path...")
                try:
                    template_list.extend(template_list_from_dataset(name))
                except FileNotFoundError:
                    warnings.warn(f"No template dataset could be found at the path {name}! Returning None. This might lead to errors in further processing.")
                    template_list.append(None)
    if is_single_template:
        return template_list[0]
    return template_list


def create_templates(main_expr: SQLColumnExpression,
                     condition_expressions: Optional[Tuple[SQLColumnExpression]] = None,
                     operators: Optional[Tuple[SQLOperator]] = None,
                     # TODO could be extended to: if main is single col description is used for first condition expression
                     # or even list of descriptions in order of occurance
                     expression_description: Optional[str] = None,  # for now only main expression
                     val_marker: str = 'val',
                     ) -> List[QuestionTemplate]:
    """ Creates Question templates with all basic condition presets given only the main expression.
        If custom condition_expressions are provided a single QuestionTemplate will be created from
        the main expression and all those conditions holding at once If for the custom conditions only
        a limited set of operators schould be applied they need to be stated explicitly,
        otherwise min, max, sum, avg, count and noop will be applied.
    """
    # different cofigurations of allowed operators depending on condition type
    arithmetic_operators = tuple([MIN, MAX, AVG, SUM])
    all_aggregators = tuple([MIN, MAX, AVG, SUM, COUNT])
    all_base_operators = tuple([MIN, MAX, AVG, SUM, NOOP, COUNT])

    # two basic condition types (all numeric, or text + numeric)
    flex_type_condition_variable_schema = {
        'condition_col': {'type': 'column',
                          'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                          },
        'condition_val': {'type': 'value',
                          'allowed_dtypes': ['numeric', 'text', 'alphanumeric']
                          },
        }
    numeric_condition_variable_schema = {
        'condition_col': {'type': 'column',
                          'allowed_dtypes': ['numeric']
                          },
        'condition_val': {'type': 'value',
                          'allowed_dtypes': ['numeric']
                          },
        }
    # definition of different basic condition setups that can be used with nearly any main expression
    condition_setups = [
        # TODO test difficulty and posthoc filter of NOOP questions (all_aggregators setting)
        ('=', all_base_operators, flex_type_condition_variable_schema),  # because NOOP results in simple lookup
        ('!=', all_base_operators, flex_type_condition_variable_schema),  # because NOOP results in simple lookup
        ('<', all_base_operators, numeric_condition_variable_schema),
        ('<=', all_base_operators, numeric_condition_variable_schema),
        ('>', all_base_operators, numeric_condition_variable_schema),
        ('>=', all_base_operators, numeric_condition_variable_schema),
        # TODO test difficulty and posthoc filter of COUNT questions (arithmetic_operators setting)
        (None, all_aggregators, {}),  # NOOP results in entire (multi-row answer) column COUNT always results in num_rows
    ]
    # ^ above stays same the following must be defined for every question type:
    # transform main expression to natural languagr template string
    main_expr_str = main_expr.generate()
    main_variables = find_template_variables(main_expr_str)
    if len(main_variables) > 1 or expression_description is not None:
        # complex column expression
        nl_main_expression = "What is the {op}" + (expression_description or f" of the expression {main_expr_str}")
    elif len(main_variables) > 0:
        # single column
        nl_main_expression = "What is the {op} of column " + "{" + main_variables[0] + "}"
    else:
        raise ValueError("Main Expression must contain at least one template variable in the format '{variable_name}'!")
    # old way of defining nl_main_expression
    #nl_main_expression = "What is the {op} of the difference between column {col1} and column {col2}<NL_CONDITION_TEMPLATE>}?"

    # define main schema (only numerical columns allowed, due to arithmethic aggregators)
    main_schema = {
        'variables': {
            var_name: {
                'type': 'column',
                'allowed_dtypes': ['numeric'],
                }
            for var_name in main_variables
        },
        'sample_strategy': 'random',
        'value_pool': 'distinct_values',
        'interpolation_args': dict()
    }

    # compute natural language template for conditions
    def _condition_to_nl(base_condition: SQLConditionTemplate) -> str:
        """Converts SQLConditionTemplate to a natural language condition phrase. """
        if isinstance(base_condition.condition_column, SQLColumnExpression):
            # complex column expression
            nl_condition_part = f"the expresion {base_condition.condition_column.generate()}"
        else:
            # single column variable
            plain_column_name = base_condition.condition_string.strip('"')
            nl_condition_part = f"column {plain_column_name}"
        if isinstance(base_condition.value, SQLColumnExpression):
            nl_value_part = f"the expression {base_condition.value.generate()}"
        else:
            nl_value_part = base_condition.value_string

        match base_condition.comparator:
            case '=': return f" given that {nl_condition_part} has a value equal to {nl_value_part}"
            case '!=': return f" given that {nl_condition_part} has a value different from {nl_value_part}"
            case '<': return f" given that {nl_condition_part} has a value greater than {nl_value_part}"
            case '<=': return f" given that {nl_condition_part} has a value greater or equal than {nl_value_part}"
            case '>': return f" given that {nl_condition_part} has a value lesser than {nl_value_part}"
            case '>=': return f" given that {nl_condition_part} has a value lesser or equal than {nl_value_part}"
            case _: return ''

    # if no custom conditions were provided proceed with the defined basic condition_setups
    if condition_expressions is None:
        question_templates = []
        for (comparator, allowed_operators, condition_schema) in condition_setups:
            if comparator is not None:
                nl_condition_template = _condition_to_nl(SQLConditionTemplate('{condition_col}', comparator, '{condition_val}'))
            else:
                nl_condition_template = ''
            #nl = nl_main_expression.replace('<NL_CONDITION_TEMPLATE>', ' given that ' + nl_condition_template)
            nl = nl_main_expression + nl_condition_template + '?'
            conditions = (
                SQLConditionTemplate(
                    '{condition_col}',
                    comparator,
                    '{condition_val}'
                    ),
                ) if comparator is not None else tuple()
            schema = copy.deepcopy(main_schema)  # reset to main schema in every iteration
            schema['variables'].update(condition_schema)  # extend schema with current condition setup
            question_templates.append(QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema))
    else:
        # a single QuestionTemplate is created -> copy main schema once and extend with all condition_expressions
        schema = copy.deepcopy(main_schema)
        nl_condition_template = ' and'.join([_condition_to_nl(condition) for condition in condition_expressions])
        nl = nl_main_expression + nl_condition_template + '?'
        # infer custom conditions' schemata
        condition_schema = {}
        for condition in condition_expressions:
            condition_str = condition.generate()
            condition_variables = find_template_variables(condition_str)
            for var_name in condition_variables:
                if val_marker in var_name:
                    var_type = 'value'
                else:
                    var_type = 'column'
                if (isinstance(condition.condition_column, SQLColumnExpression)
                        or isinstance(condition.value, SQLColumnExpression)
                        or condition.comparator in ('<', '<=', '>', '>=')):
                    allowed_dtypes = ['numeric']
                else:
                    allowed_dtypes = ['numeric', 'text', 'alphanumeric']
                condition_schema.update({var_name: {'type': var_type, 'allowed_dtypes': allowed_dtypes}})
        schema['variables'].update(condition_schema)
        question_templates = [QuestionTemplate(nl, main_expr, (operators or all_base_operators), conditions, schema)]
        # conditions = condition expressions <- can be inserted directly in Question templates
    return question_templates


def get_standard_templates(save_path: Optional[str] = None, cache_dir: str = '/home/mamba/.cache') -> List[QuestionTemplate]:
    try:
        coalesce_save_path = save_path or cache_dir + '/templates/standard_templates'
        # TODO implement logging instead of print
        #logger.info(f"Try Loading standard templates from {save_path}...")
        print(f"Try Loading standard templates from {coalesce_save_path} ...")
        return template_list_from_dataset(coalesce_save_path)
    except FileNotFoundError:
        # TODO implement logging instead of print
        #logger.info("No saved version was found. Creating standard templates...")
        print("No saved version was found. Creating standard templates...")
        basic_templates = create_templates(
            SQLColumnExpression(("{col1}",)),
            expression_description=" of column {col1}"
            )
        diff_templates = create_templates(
            SQLColumnExpression(("{col1}", "{col2}"), operand='-'),
            expression_description=" of the difference between column {col1} and column {col2}"
            )
        ratio_templates = create_templates(
            SQLColumnExpression(("{col1}", "{col2}"), operand='/'),
            expression_description=" of the ratio of column {col1} to column {col2}"
            )
        """ nicer expression below
        expression_templates = create_templates(
            SQLColumnExpression(
                (
                    SQLColumnExpression(
                        (SQLColumnExpression(("{col1}", "{col2}"), operand='*'),
                        SQLColumnExpression(("{col1}", "{col2}"), operand='+'),
                        ),
                        operand='-',
                        ),
                    SQLColumnExpression(
                        (SQLColumnExpression(("{col1}", "{col1}"), operand='*'),
                        SQLColumnExpression(
                            (SQLColumnExpression(("{col2}", "{col2}"), operand='*'),
                            "{col2}"
                            ),
                            operand='-',
                            ),
                        ),
                        operand='+',
                        )
                    ),
                operand='/',
                ),
            )
            """
        expression_templates = create_templates(
            SQLColumnExpression(
                ("{col1}",
                 SQLColumnExpression(
                     (SQLColumnExpression(("{col1}", "{col2}"), operand='-'),
                      SQLColumnExpression(
                          (SQLColumnExpression((2, SQLColumnExpression(("{col1}", "{col2}"), operand='*')), operand='*'),
                           SQLColumnExpression(
                               (SQLColumnExpression(("{col1}", "{col1}"), operand='*'),
                                SQLColumnExpression(("{col2}", "{col2}"), operand='*')
                                ),
                               operand='+'
                               )
                           ),
                          operand='/'
                          )
                      ),
                     operand='*'
                     )
                 ),
                operand='*'
                )
            )
        templates = [template
                     for spec in [basic_templates, diff_templates, ratio_templates, expression_templates]
                     for template in spec
                     ]
        if save_path:
            dataset = datasets.Dataset.from_list([template.to_state_dict() for template in templates])
            dataset.save_to_disk(save_path)
        return templates


def create_dataset(templates: Union[QuestionTemplate, List[QuestionTemplate]],
                   dataset_name: str,
                   description: Optional[str] = None,
                   table_corpus: str = 'wikitables',
                   split: Optional[str] = None,
                   tables: Optional[Union[List[Table], datasets.Dataset]] = None,
                   save: bool = True,
                   args: Optional[DataProcessingArgs] = None,
                   ) -> TableQuestionDataSet:
    if args is None:
        args = DataProcessingArgs()  # get default args
    if isinstance(templates, QuestionTemplate):
        templates = [templates]  # make sure templates is a list
    # load table corpus
    if tables is not None:
        table_corpus = 'custom_' + table_corpus
    else:
        tables = load_table_dataset(table_corpus, split, args.cache_dir)
    if len(tables) == 0:
        warnings.warn("Empty taple corpus encountered. Nothing to generate!")
    save_name = f"{table_corpus}_{dataset_name}_{split or 'all'}"
    dataset = TableQuestionDataSet(
        save_name,
        description=description,
        question_templates=templates,
        tables=tables,
        num_proc=args.num_proc,
        max_num_value_samples=args.max_num_value_samples,
        max_value_length=args.max_value_length,
        max_questions_per_table=args.max_questions_per_table,
        load_from_cache=args.load_from_cache,
        delete_intermediate_cache=args.delete_intermediate_cache,
        )
    if save:
        save_version(dataset, args.cache_dir, save_name)
    return dataset


def create_basic_table_question_dataset(tables,
                                        args: DataProcessingArgs,
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
                                       tables=tables,
                                       )
        # get cache path of tables to not save tables multiple times (only trable_id will be saved); if tables are in-memory save as new table dataset
        fallback_table_dataset_path = cache_path + '/' + cache_file_name + '_tables'
        table_dataset_path = ensure_table_dataset_on_disk(tables, save_path=fallback_table_dataset_path)
        if save:
            save_version(dataset, cache_path, cache_file_name, questions_only=True, table_dataset_save_path=table_dataset_path)
    return dataset


def create_table_question_dataset(tables: Union[datasets.Dataset, List[Table]],
                                  template: Union[QuestionTemplate, List[QuestionTemplate]],
                                  args: DataProcessingArgs,
                                  name: Optional[str] = None,
                                  description: Optional[str] = None,
                                  questions_only: bool = False,
                                  save_path: Optional[str] = None,
                                  ) -> TableQuestionDataSet:
    if name is None:
        num_templates = len(template) if isinstance(template, list) else 1
        num_conditions = template.num_conditions if isinstance(template, QuestionTemplate) else '?'
        main_expr = template.main_expression.generate() if isinstance(template, QuestionTemplate) else '?'
        auto_name = f"{len(tables)}_tables_{num_templates}_templates_{main_expr}_{num_conditions}_conditions_{datetime.now().strftime('%y%m%d_%H%M_%S_%f')}"
    dataset = TableQuestionDataSet(name or auto_name,
                                   description=description,
                                   question_templates=[template] if isinstance(template, QuestionTemplate) else template,
                                   tables=tables,
                                   )
    if questions_only:
        # get cache path of tables to not save tables multiple times (only trable_id will be saved); if tables are in-memory save as new table dataset
        fallback_table_dataset_path = save_path or ('./' + (name or auto_name)) + '_tables'
        table_dataset_path = ensure_table_dataset_on_disk(tables, save_path=fallback_table_dataset_path)
    if save_path:
        save_version(dataset, save_path, questions_only=questions_only, table_dataset_save_path=table_dataset_path if questions_only else None)
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


def create_add_one_question_dataset(tables,
                                    name='wikitables_test',
                                    use_cache: bool = True,
                                    cache_path: str = './data/NumTabQA/.cache',
                                    save=True,
                                    ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_add_one_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg, sum or no operation combined with a simple value lookup condition of a different column and adding one to every numerical value.
            Using WikiTables test set.

            """
        nl = "What is the {op} of column {col1} after adding one to every value given that {col2} has value {val1}?"
        main_expr = SQLColumnExpression(("{col1} + 1",))
        conditions = (SQLConditionTemplate('{col2}', '=', '{val1}'),)  # TODO think about condition type: numerical >/<=; text =; numerical + text =
        allowed_operators = tuple([MIN, MAX, AVG, SUM, NOOP, COUNT])
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
        add_one_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_add_one',
                                       description=base_description,
                                       question_templates=[add_one_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_diff_table_question_dataset(tables,
                                       name='wikitablequestions_test_diff',
                                       use_cache: bool = True,
                                       cache_path: str = './data/NumTabQA/.cache',
                                       save=True,
                                       ) -> TableQuestionDataSet:
    if use_cache:
        dataset = caching(name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum of the difference between two numeric columns combined with a simple value lookup condition.
            Using WikiTables test set.

            """
        nl = "What is the {op} of the difference between column {col1} and {col2} given that {col3} has value {val1}?"
        main_expr = SQLColumnExpression(("{col1}", "{col2}"), operand='-')
        conditions = (SQLConditionTemplate('{col3}', '=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, COUNT, NOOP])
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col3': {'type': 'column',
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
        dataset = TableQuestionDataSet(name + '_diff',
                                       description=base_description,
                                       question_templates=[basic_template],
                                       tables=tables,
                                       compute_coordinates=False,
                                       )
        if save:
            save_version(dataset, cache_path, name)
    return dataset


def create_difference_plus_geq_question_dataset(tables,
                                                name='wikitables_test',
                                                use_cache: bool = True,
                                                cache_path: str = './data/NumTabQA/.cache',
                                                save=True,
                                                ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_diff_geq_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg, sum or no operation over the difference of two numerical columns combined with a greater or equal condition of a first column (minuend).
            Using WikiTables test set.

            """
        nl = "What is the {op} of the difference between column {col1} and column {col2} given that {col1} has a value of at least {val1}?"
        main_expr = SQLColumnExpression(("{col1} - {col2}",))
        conditions = (SQLConditionTemplate('{col1}', '>=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, NOOP, COUNT])
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'val1': {'type': 'value',
                         'allowed_dtypes': ['numeric']
                         }
            },
            'sample_strategy': 'random',
            'value_pool': 'distinct_values',
            'interpolation_args': dict()
        }
        diff_geq_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_diff_geq',
                                       description=base_description,
                                       question_templates=[diff_geq_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_ratio_question_dataset(tables,
                                  name='wikitables_test',
                                  use_cache: bool = True,
                                  cache_path: str = './data/NumTabQA/.cache',
                                  save=True,
                                  ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_ratio_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum over the ratio of two numerical columns.
            Using WikiTables test set.

            """
        nl = "What is the {op} of the ratio between column {col1} and column {col2}?"
        main_expr = SQLColumnExpression(("{col1} / {col2}",))
        conditions = tuple()
        allowed_operators = tuple([MIN, MAX, AVG, SUM])
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
            },
            'sample_strategy': 'random',
            'value_pool': 'distinct_values',
            'interpolation_args': dict()
        }
        ratio_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_ratio',
                                       description=base_description,
                                       question_templates=[ratio_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def create_expression_dataset(tables,
                              name='wikitablequestions_test_expression',
                              use_cache: bool = True,
                              cache_path: str = './data/NumTabQA/.cache',
                              save=True,
                              memory_mapped=True,
                              ) -> TableQuestionDataSet:
    if use_cache:
        dataset = caching(name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum over a complex arithmetic expression of two numerical columns combined with a simple value lookup condition.
            Using WikiTableQuestions.

            """
        nl = "What is the {op} of the expression  between column {col1} and column {col2}?"
        main_expr = SQLColumnExpression(
            (SQLColumnExpression(
                (SQLColumnExpression(("{col1}", "{col2}"), operand='*'),
                 SQLColumnExpression(("{col1}", "{col2}"), operand='+'),
                 ),
                operand='-',
                ),
             SQLColumnExpression(
                 (SQLColumnExpression(("{col1}", "{col1}"), operand='*'),
                  SQLColumnExpression(
                      (SQLColumnExpression(("{col2}", "{col2}"), operand='*'),
                       "{col2}"
                       ),
                      operand='-',
                      ),
                  ),
                 operand='+',
                 )
             ),
            operand='/',
            )
        # string representation TODO funtion for automated parsing from string
        #'(("{col1}" * "{col2}") - ("{col1}" + "{col2}")) / ("{col1}" * "{col1}") + (("{col2}" * "{col2}") - "{col2}")'
        conditions = (SQLConditionTemplate('{col3}', '=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, NOOP])  # no count as same as in other datasets
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col3': {'type': 'column',
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
        ratio_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_expression',
                                       description=base_description,
                                       question_templates=[ratio_template],
                                       tables=tables,
                                       compute_coordinates=False,
                                       memory_mapped=memory_mapped,
                                       )
        if save:
            save_version(dataset, cache_path, name)
    return dataset


def create_expression_dataset2(tables,
                               name='wikitablequestions_test_expression',
                               use_cache: bool = True,
                               cache_path: str = './data/NumTabQA/.cache',
                               save=True,
                               memory_mapped=True,
                               ) -> TableQuestionDataSet:
    if use_cache:
        dataset = caching(name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum over a complex arithmetic expression of two numerical columns combined with a simple value lookup condition.
            Using WikiTableQuestions.

            """
        nl = "What is the {op} of the expression  between column {col1} and column {col2}?"
        main_expr = SQLColumnExpression(
            (SQLColumnExpression(
                (SQLColumnExpression(("{col1}", "{col2}"), operand='*'),
                 SQLColumnExpression(("{col1}", "{col2}"), operand='+'),
                 ),
                operand='-',
                ),
             SQLColumnExpression(
                 (SQLColumnExpression(("{col1}", "{col1}"), operand='*'),
                  SQLColumnExpression(
                      (SQLColumnExpression(("{col2}", "{col2}"), operand='*'),
                       "{col2}"
                       ),
                      operand='-',
                      ),
                  ),
                 operand='+',
                 )
             ),
            operand='/',
            )
        # string representation TODO funtion for automated parsing from string
        #'(("{col1}" * "{col2}") - ("{col1}" + "{col2}")) / ("{col1}" * "{col1}") + (("{col2}" * "{col2}") - "{col2}")'
        conditions = (SQLConditionTemplate('{col3}', '=', '{val1}'),)
        allowed_operators = tuple([MIN, MAX, AVG, SUM, NOOP])  # no count as same as in other datasets
        schema = {
            'variables': {
                'col1': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col2': {'type': 'column',
                         'allowed_dtypes': ['numeric']
                         },
                'col3': {'type': 'column',
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
        ratio_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_expression',
                                       description=base_description,
                                       question_templates=[ratio_template],
                                       tables=tables,
                                       compute_coordinates=False,
                                       memory_mapped=memory_mapped,
                                       )
        if save:
            save_version(dataset, cache_path, name)
    return dataset


def create_postprocessed_versions(data_version_function: Callable[..., TableQuestionDataSet] = create_basic_table_question_dataset,
                                  name: str = 'basic',
                                  table_corpus='wikitablequestions',
                                  splits=('test', 'train', 'validation'),
                                  remove_multi_answer: bool = True,
                                  single_row_agg_tolerances: Tuple[float] = (0.2, 0.0),
                                  cache_path: str = './data/NumTabQA/.cache',
                                  ):
    for split in splits:
        table_dataset = create_table_dataset(base_dataset_name=table_corpus, base_dataset_split=split, use_cache=True)
        base_name = f"{table_corpus}_{split}_{name}"
        dataset = data_version_function(table_dataset, name=base_name, use_cache=False)
        apply_quality_filters(dataset,
                              remove_multi_answer=remove_multi_answer,
                              single_row_agg_tolerances=single_row_agg_tolerances,
                              dataset_name=base_name,
                              cache_path=cache_path
                              )


def apply_filter_condition(dataset: datasets.Dataset, num_proc: Optional[int] = 4) -> datasets.Dataset:
    """ Applies a precomputed filter condition to a dataset that has equal length sequences as fields.
        filter_condition already needs to be a field in dataset and contain a list of booleans (True is kept False is filtered).
        The length of the list must be equivalent to every sequence field in the dataset.
        Only sequence entries where the corresponding filter_condition is True are kept for each field.
        The field filter_condition is removed from the dataset during the process.
    """
    if 'filter_condition' not in dataset.column_names:
        raise ValueError("Dataset must contain a 'filter_condition' field with a list of booleans!")
    return dataset.map(lambda x: {col_name: [x[col_name][i]
                                             for i, is_kept in enumerate(x['filter_condition'])
                                             if is_kept
                                             ]
                                  for col_name in x.keys()
                                  if isinstance(x[col_name], list)
                                  },
                       desc="Applying pre-computed filter condition...",
                       remove_columns=['filter_condition'],
                       num_proc=num_proc,
                       )


def apply_quality_filters(dataset: Union[datasets.Dataset, TableQuestionDataSet],
                          remove_multi_answer: bool = True,
                          single_row_agg_tolerances: Tuple[float] = (0.2, 0.0),
                          threshold: int = 2,
                          save: bool = True,
                          dataset_name: Optional[str] = None,
                          num_proc: Optional[int] = 12,
                          cache_path: str = '/home/mamba/.cache',
                          ):
    if save and dataset_name is None:
        # TODO infer from cache files
        raise ValueError("Need to provide dataset name if you want to save it!")
    # remove invalid answers
    if isinstance(dataset, datasets.Dataset):
        dataset = dataset.map(lambda x: {'filter_condition': [x['answers'][i].lower() not in ['', 'none', 'nan', 'n/a']
                                                              for i in range(len(x['questions']))
                                                              ]
                                         },
                              desc="Prepare filter_condition: valid answer string...",
                              num_proc=num_proc,
                              )
        dataset = apply_filter_condition(dataset, num_proc=num_proc)
    else:
        dataset._remove_unanswered_questions()

    # remove multi-answer questions
    if remove_multi_answer:
        if isinstance(dataset, datasets.Dataset):
            dataset = dataset.map(lambda x: {'filter_condition': [x['is_multy_row_answer'][i].lower() == 'false'
                                                                  if isinstance(x['is_multy_row_answer'][i], str)
                                                                  else x['is_multy_row_answer'][i] is False
                                                                  for i in range(len(x['questions']))
                                                                  ]
                                             },
                                  desc="Prepare filter_condition: multi_answer questions...",
                                  num_proc=num_proc,
                                  )
            dataset = apply_filter_condition(dataset, num_proc=num_proc)
        else:
            dataset.remove_multi_answer_questions()
        if save:
            save_version(dataset, cache_path, dataset_name + '_filtered_multi_answer')

    # remove questions with low aggregation count
    old_dataset = dataset  # keep reference to original dataset for subsequent removal of questions with lower tolerance
    for tolerance in sorted(single_row_agg_tolerances, reverse=True):  # filter higher tolerances first subsequent removal <- TODO check if this is still necessary
        if isinstance(dataset, TableQuestionDataSet):
            # TODO revisit if in-place removal makes sense in this context with subsequent tolerances
            dataset.remove_questions_with_lower_aggregation_count(threshold=threshold, tolerance=tolerance)
        else:
            if not isinstance(dataset, datasets.Dataset):
                raise ValueError(f"Dataset must be either a TableQuestionDataSet or a datasets.Dataset (not {type(dataset)})!")
            # Analogous to remove_questions_with_lower_aggregation_count() from TableQuestionDataSet just for post hock huggingface datasets serialization
            if tolerance > 1.0 or tolerance < 0.0:
                raise ValueError(f"tolerance must be between 0 and 1 but was {tolerance}!"
                                 "It represents the allowed proportion of questions with aggregation of rows with less than threshold.")
            if tolerance == 0.0:
                dataset = dataset.map(lambda x: {'filter_condition': [x['aggregators'][i] == '' or int(x['aggregation_num_rows'][i] or -1) >= threshold
                                                                      for i in range(len(x['questions']))
                                                                      ]
                                                 },
                                      desc=f"Prepare filter_condition: questions with agg_count lower {threshold}...",
                                      num_proc=num_proc,
                                      )
                dataset = apply_filter_condition(dataset, num_proc=num_proc)
            else:
                warnings.warn("For datasets.Dataset serialization the tolerance is approximated through probabalistic sampling. "
                              "This may lead minor shifts in the data distribution compared to the deterministic case.")
                dataset = old_dataset.map(lambda x: {'filter_condition': [int(x['aggregation_num_rows'][i] or -1) >= threshold
                                                                          or x['aggregators'][i] == ''
                                                                          or np.random.rand() <= tolerance
                                                                          for i in range(len(x['questions']))
                                                                          ],
                                                     },
                                          desc=f"Prepare filter_condition: questions with agg_count lower {threshold} (tolerance {tolerance})...",
                                          num_proc=num_proc,
                                          )
                dataset = apply_filter_condition(dataset, num_proc=num_proc)
        if save:
            save_version(dataset, cache_path, dataset_name + f"{'_filtered_multi_answer' if remove_multi_answer else ''}_filter_agg_count_{int(tolerance*100)}")
    return dataset


def main(single_version: bool = True, table_corpus: str = 'wikitablequestions', split: Optional[str] = None, _skip_first: Optional[int] = None):
    # use pre-configured functiun for dataset (single dataset)
    if single_version:
        #create_postprocessed_versions()#data_version_function=create_diff_table_question_dataset, name='diff')  # TODO compare diff with interactive version
        create_postprocessed_versions(data_version_function=create_expression_dataset, name='expression')
        return
    # interactive (API) version for on-the-fly creation
    main_expressions = []
    dataset_names = []
    descriptions = []
    expression_descriptions = []
    # single col
    main_expressions.append(SQLColumnExpression('{col1}'))
    dataset_names.append('filtered_col')
    descriptions.append('Basic dataset with an aggregator over a single column combined with all standard filter conditions.')
    expression_descriptions.append(None)
    # diff
    main_expressions.append(SQLColumnExpression(('{col1}', '{col2}',), '-'))
    dataset_names.append('difference')
    descriptions.append('Aggregator over difference between two columns combined with all standard filter conditions.')
    expression_descriptions.append(None)
    # ratio
    main_expressions.append(SQLColumnExpression(('{col1}', '{col2}',), '/'))
    dataset_names.append('ratio')
    descriptions.append('Aggregator over ratio between two columns combined with all standard filter conditions.')
    expression_descriptions.append(None)
    # complex expression one
    main_expressions.append(SQLColumnExpression(
        (
            SQLColumnExpression(('{col1}', '{col2}',), '*'),
            SQLColumnExpression(
                (
                    SQLColumnExpression(('{col1}', '{col1}',), '*'),
                    SQLColumnExpression(('{col2}', '{col2}',), '*'),
                    ),
                '+'
                ),
            ),
        '/'
        )
    )
    dataset_names.append('complex_expression_1')
    descriptions.append('Aggregator over complex arithmetic expression (x*y/x^2+y^2) combined with all standard filter conditions.')
    expression_descriptions.append(None)
    # add one
    main_expressions.append(SQLColumnExpression(('{col1}', '1',), '+'))
    dataset_names.append('add_one')
    descriptions.append('Aggregator over single column after adding a constant of one to it combined with all standard filter conditions.')
    expression_descriptions.append("of column {col1} after adding one to every value")

    if split is None:
        # TODO determine splits of dataset
        splits = []
    else:
        splits = [split]
    for split_name in splits:
        for i, (main_expr, dataset_name, description, expression_description) in enumerate(zip(main_expressions, dataset_names, descriptions, expression_descriptions)):
            if _skip_first > i:
                continue
            template_list = create_templates(main_expr, expression_description=expression_description)
            dataset = create_dataset(template_list,
                                     dataset_name=dataset_name,
                                     description=description,
                                     table_corpus=table_corpus,
                                     split=split_name,
                                     )
            apply_quality_filters(dataset)
            print(f'done with {dataset_name} split {split}')  # TODO proper logging instead


def template_list_from_dataset(dataset: Union[datasets.Dataset, str, PurePath]) -> List[QuestionTemplate]:
    if isinstance(dataset, (str, PurePath)):
        dataset = datasets.Dataset.load_from_disk(dataset)
    return [QuestionTemplate.from_state_dict(template) for template in dataset]


def process_template_names(template_names: Union[str, List[str]]) -> str:
    if isinstance(template_names, str):
        template_names = [template_names]  # always make it a list
    processed_names = []
    for name in template_names:
        # check if name is a path and extract name
        if '/' in name:
            processed_names.append(PurePath(name).name)  # if saved with save_version should be .parent.name because of timestamp
        else:
            processed_names.append(name)
    return '_'.join(processed_names)


def generate_questions_from_templates(templates: Union[datasets.Dataset, str, List[str], PurePath],
                                      table_corpus: str,
                                      dataset_splits: Union[str, List[str]] = 'test',
                                      dataset_name: Optional[str] = None,
                                      dataset_description: str = '',
                                      args: Optional[DataProcessingArgs] = None,
                                      ) -> Union[datasets.Dataset, Dict[str, datasets.Dataset]]:
    if args is None:
        # TODO think if also complete initialization from args is possible
        args = DataProcessingArgs()  # use default arguments
    # type check fpr templates
    if not isinstance(templates, (datasets.Dataset, str, list, PurePath)):
        raise ValueError(f"templates must be either a datasets.Dataset, a PurePath, a string or a list of strings but {type(templates)} was passed!")
    # make sure dataset_splits is a list
    if isinstance(dataset_splits, str):
        dataset_splits = [dataset_splits]
    # determine dataset_name if not provided
    if dataset_name is None:
        if isinstance(templates, (str, list)):
            dataset_name = process_template_names(templates)
        else:
            if isinstance(templates, PurePath):
                templates = datasets.load_from_disk(templates)  # TODO use caching
            # from here on templates should be a datasets.Dataset (either loaded from PurePath or passed as datasets.Dataset directly)
            dataset_name = str(PurePath(templates.cache_files[0]).parent.name)  # if saved with save_version should be .parent.parent.name because of timestamp
    # get template list from name or Dataset
    if isinstance(templates, str):
        template_list = get_template_by_name([templates])  # make sure putput is a list, by passing a list
    elif isinstance(templates, list):
        template_list = get_template_by_name(templates)
    else:  # either PurePath (if dataset_name was passed not as None) or datasets.Dataset (either passed explicitly or loaded in dataset_name resolution)
        template_list = template_list_from_dataset(templates)

    split_dict = {}
    for split in dataset_splits:
        split_dict[split] = create_dataset(template_list,
                                           dataset_name=dataset_name,
                                           description=dataset_description,
                                           table_corpus=table_corpus,
                                           split=split,
                                           save=True,
                                           args=args,
                                           )
    if len(dataset_splits) == 1:
        return split_dict[dataset_splits[0]]
    return split_dict


if __name__ == "__main__":
    # TODO refactor functions should do one thing (e.g apply quality filter should not save dataset version) -> have script instead that only executes all required functions
    # TODO separate template creation and dataset creation modules
    args = dargparse(DataProcessingArgs)

    generate_questions_from_templates(templates=args.template_names,  #template_path or args.cache_dir + '/templates/' + template_name,
                                      table_corpus=args.table_corpus,
                                      dataset_splits=args.splits,
                                      dataset_name=args.dataset_name,
                                      dataset_description='All combinations of standard templates with the gittables_group_filtered corpus.',
                                      args=args,
                                      )

    # apply quality filters and save filtered versions
    for split in args.splits:
        dataset_name = f"{args.table_corpus}_{args.dataset_name or process_template_names(args.template_names)}_{split}"
        dataset = caching(dataset_name, cache_path=args.cache_dir)
        apply_quality_filters(dataset, dataset_name=dataset_name, cache_path=args.cache_dir, save=True)
    #main()
    # gittables example
    # table_dataset = load_table_dataset(table_corpus='gittables_subset_10', split='train', cache_path='/home/mamba/.cache')
    # create_basic_table_question_dataset(table_dataset, name='gittables_subset_10_train', use_cache=False, cache_path='/home/mamba/.cache')
