import copy
import warnings

from typing import Optional, Tuple, List

from numerical_table_questions.data_caching import save_version, caching
from numerical_table_questions.data_synthesis import QuestionTemplate, Table, TableQuestionDataSet, load_table_dataset
from numerical_table_questions.sql_templates import (
    SQLColumnExpression, SQLConditionTemplate, SQLOperator,
    MIN, MAX, SUM, AVG, COUNT, NOOP, find_template_variables,
)


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
        ('=', all_base_operators, flex_type_condition_variable_schema),
        ('!=', all_aggregators, flex_type_condition_variable_schema),
        ('<', all_aggregators, numeric_condition_variable_schema),
        ('<=', all_aggregators, numeric_condition_variable_schema),
        ('>', all_aggregators, numeric_condition_variable_schema),
        ('>=', all_aggregators, numeric_condition_variable_schema),
        (None, arithmetic_operators, {}),
    ]
    # ^ above stays same the following must be defined for every question type:
    # transform main expression to natural languagr template string
    main_expr_str = main_expr.generate()
    main_variables = find_template_variables(main_expr_str)
    if len(main_variables) > 1:
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
            nl_condition_part = f"column {base_condition.generate(in_parts=True)['condition_column'].strip('"')}"
        if isinstance(base_condition.value, SQLColumnExpression):
            nl_value_part = f"the expression {base_condition.value.generate()}"
        else:
            nl_value_part = base_condition.generate(in_parts=True)['value']
        # old behavior (throw error # TODO delete)
        #if isinstance(base_condition.condition_column, SQLColumnExpression) or isinstance(base_condition.value, SQLColumnExpression):
        #    raise ValueError("The provided base_condition contains SQLColumnExpression(s)! "
        #                     "This function only works for basic conditions with a simple condition column variable and a value variable. ")
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


def create_difference_question_dataset(tables,
                                       name='wikitables_test',
                                       use_cache: bool = True,
                                       cache_path: str = './data/NumTabQA/.cache',
                                       save=True,
                                       ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_diff_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum over the difference of two numerical columns.
            Using WikiTables test set.

            """
        nl = "What is the {op} of the difference between column {col1} and column {col2}?"
        main_expr = SQLColumnExpression(("{col1} - {col2}",))
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
        diff_template = QuestionTemplate(nl, main_expr, allowed_operators, conditions, schema)
        dataset = TableQuestionDataSet(name + '_diff',
                                       description=base_description,
                                       question_templates=[diff_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
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
                              name='wikitables_test',
                              use_cache: bool = True,
                              cache_path: str = './data/NumTabQA/.cache',
                              save=True,
                              ) -> TableQuestionDataSet:
    cache_file_name = f"{name}_expression_dataset"
    if use_cache:
        dataset = caching(cache_file_name, cache_path=cache_path)
    else:
        base_description = \
            """
            Basic SQL operators min, max, avg or sum over a complex arithmetic expression of two numerical columns.
            Using WikiTables test set.

            """
        nl = "What is the {op} of the expression  between column {col1} and column {col2}?"
        main_expr = SQLColumnExpression(("(({col1} * {col2}) - ({col1} + {col2})) / ({col1} * {col1}) + (({col2} * {col2}) - {col2})",))
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
        dataset = TableQuestionDataSet(name + '_expression',
                                       description=base_description,
                                       question_templates=[ratio_template],
                                       tables=tables
                                       )
        if save:
            save_version(dataset, cache_path, cache_file_name)
    return dataset


def main(table_corpus: str = 'wikitablequestions', split: Optional[str] = None, _skip_first: Optional[int] = None):
    main_expressions = []
    dataset_names = []
    descriptions = []
    expression_descriptions = []
    # single col
    main_expressions.append(SQLColumnExpression('{col1}'))
    dataset_names.append('filtered_col')
    descriptions.append('')
    expression_descriptions.append(None)
    # diff
    main_expressions.append(SQLColumnExpression(('{col1}', '{col2}',), '-'))
    dataset_names.append('difference')
    descriptions.append('')
    expression_descriptions.append(None)
    # ratio
    main_expressions.append(SQLColumnExpression(('{col1}', '{col2}',), '/'))
    dataset_names.append('ratio')
    descriptions.append('')
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
    descriptions.append('')
    expression_descriptions.append(None)
    # add one
    main_expressions.append(SQLColumnExpression(('{col1}', '1',), '+'))
    dataset_names.append('add_one')
    descriptions.append('')
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
            print(f'done with {dataset_name} split {split}')  # TODO proper logging instead


if __name__ == "__main__":
    main()
