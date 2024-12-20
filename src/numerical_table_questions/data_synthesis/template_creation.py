import copy
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePath

import datasets
from typing import Optional, Tuple, List, Union

from numerical_table_questions.arguments import DataProcessingArgs
from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.data_synthesis.dataset import TableQuestionDataSet, ensure_table_dataset_on_disk
from numerical_table_questions.data_synthesis.question_template import QuestionTemplate
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.table_creation import create_table_dataset, load_table_dataset
from numerical_table_questions.data_utils import get_cache_path
from numerical_table_questions.sql_templates import (
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
            case _:
                warnings.warn("No template could be found for the name {name}! Trying to find template dataset at this path.")
                #raise ValueError("No template could be found for the name {name}! Please make sure you have crafted and registered the template correctly.")
                template_list = None if not is_single_template else [None]
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

    # TODO think of number of samples per group
    # TODO rethink if excluding operators is necessary (post-hoc filters take care of it?)
    # definition of different basic condition setups that can be used with nearly any main expression
    condition_setups = [
        ('=', all_aggregators, flex_type_condition_variable_schema),
        ('!=', all_aggregators, flex_type_condition_variable_schema),
        ('<', all_base_operators, numeric_condition_variable_schema),
        ('<=', all_base_operators, numeric_condition_variable_schema),
        ('>', all_base_operators, numeric_condition_variable_schema),
        ('>=', all_base_operators, numeric_condition_variable_schema),
        (None, arithmetic_operators, {}),
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


def get_standard_templates(save_path: Optional[str] = None) -> List[QuestionTemplate]:
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


def create_dataset(template_list,
                   dataset_name: str,
                   description: Optional[str] = None,
                   table_corpus: str = 'wikitables',
                   split: Optional[str] = None,
                   tables: Optional[List[Table]] = None,
                   cache_path: str = './data/NumTabQA/.cache',
                   save: bool = True,
                   ) -> TableQuestionDataSet:
    # load table corpus
    if tables is not None:
        table_corpus = 'custom_' + table_corpus
    else:
        tables = load_table_dataset(table_corpus, split, cache_path)
    if len(tables) == 0:
        warnings.warn("Empty taple corpus encountered. Nothing to generate!")
    save_name = f"{table_corpus}_{dataset_name}"
    dataset = TableQuestionDataSet(
        save_name,
        description=description,
        question_templates=[template_list],
        tables=tables
        )
    if save:
        save_version(dataset, cache_path, save_name)
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
                                       args,
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
                                   args,
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


def apply_quality_filters(dataset,
                          remove_multi_answer: bool = True,
                          single_row_agg_tolerances: Tuple[float] = (0.2, 0.0),
                          save: bool = True,
                          dataset_name: Optional[str] = None,
                          cache_path: str = './data/NumTabQA/.cache',
                          ):
    if save and dataset_name is None:
        raise ValueError("Need to provide dataset name if you want to save it!")
    if remove_multi_answer:
        dataset.remove_multi_answer_questions()
        if save:
            save_version(dataset, cache_path, dataset_name + '_filtered_multi_answer')
    for tolerance in sorted(single_row_agg_tolerances, reverse=True):  # filter higher tolerances first subsequent removal
        dataset.remove_questions_with_lower_aggregation_count(tolerance=tolerance)
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
        dataset = datasets.load_from_disk(dataset)
    return [QuestionTemplate.from_state_dict(template) for template in dataset]


def generate_questions_from_templates(templates: Union[datasets.Dataset, str, PurePath],
                                      table_corpus: str,
                                      dataset_splits: Union[str, List[str]] = 'test',
                                      dataset_name: Optional[str] = None,
                                      dataset_description: str = '',
                                      cache_path: str = './data/NumTabQA/.cache',
                                      ) -> None:
    if isinstance(dataset_splits, str):
        dataset_splits = [dataset_splits]  # make sure it is a list
    if dataset_name is None:
        # load template dataset to extract name from cache path
        if isinstance(templates, (str, PurePath)):
            templates = datasets.load_from_disk(templates)
        dataset_name = (table_corpus +
                        str(PurePath(templates.cache_files[0]).parent.name)  # if saved with save_version should be .parent.parent.name because of timestamp
                        )
    # if dataset_name argument was None (and templates argument was originally str or Path) templates will already be of type datasets.Dataset
    # and does not need to be loaded by template_list_from_dataset again
    template_list = template_list_from_dataset(templates)
    for split in dataset_splits:
        dataset = create_dataset(template_list,
                                 dataset_name=dataset_name,
                                 description=dataset_description,
                                 table_corpus=table_corpus,
                                 split=split,
                                 )
        save_version(dataset, cache_path=cache_path, dataset_name=dataset_name + f'_{split}')



if __name__ == "__main__":
    # TODO refactor functions should do one thing (e.g apply quality filter should not save dataset version) -> have script instead that only executes all required functions
    # TODO separate template creation and dataset creation modules
    table_corpus = 'wikitablequestions'
    dataset_splits = ['test', 'train', 'validation']
    template_set_name = 'standard_templates'
    cache_path = './data/NumTabQA/.cache'
    generate_questions_from_templates(templates='./data/NumTabQA/.cache/templates/standard_templates',
                                      table_corpus=table_corpus,
                                      dataset_splits=dataset_splits,
                                      dataset_name=table_corpus + '_' + template_set_name,
                                      dataset_description='All combinations of standard templates with the wikitablequestion corpus.',
                                      cache_path=cache_path)
    for split in dataset_splits:
        dataset_name = f"{table_corpus}_{template_set_name}_{split}"
        dataset = caching(dataset_name, cache_path=cache_path)
        apply_quality_filters(dataset, dataset_name=dataset_name, cache_path=cache_path, save=True)
    #main()
    # gittables example
    # table_dataset = load_table_dataset(table_corpus='gittables_subset_10', split='train', cache_path='/home/mamba/.cache')
    # create_basic_table_question_dataset(table_dataset, name='gittables_subset_10_train', use_cache=False, cache_path='/home/mamba/.cache')
