import hashlib
from typing import Dict, List, Optional, Tuple, Type, Union

import datasets

from numerical_table_questions.data_caching import delete_dataset
from numerical_table_questions.data_synthesis import Table, TableQuestion, QuestionTemplate
from numerical_table_questions.sql_templates import SQLTemplate, SQLOperatorTemplate, SQLOverClauseTemplate


def create_all_question_fields(sample,
                               template_obj: QuestionTemplate,
                               create_alternatives: bool = False,
                               do_count_augmentation:bool = True,
                               ):
    table = Table.from_state_dict(sample['table'])
    generated_questions = []
    generated_count_questions = []
    used_datatypes = [template_obj._schema['variables'][variable]['allowed_dtypes']
                        for variable in template_obj._schema['variables'].keys()
                        if template_obj._schema['variables'][variable]['type'] == 'value'
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
    for op_idx, operator in enumerate(template_obj._operators):
        # create Template object and generate template string for current operator (aggregator)
        sql_template_obj = SQLTemplate(
            SQLOperatorTemplate(
                operator.sql,
                template_obj.main_expression,
                brackets=None,
                over_clause=SQLOverClauseTemplate(
                    operator.sql_partition_cols,
                    operator.sql_order_cols,
                    operator.sql_frame_size
                ) if operator.sql_over_clause else None
            ),
            conditions=template_obj.conditions
        )
        sql_template = sql_template_obj.generate()

        # aggregator configurations with same allowed dtypes draw samples from the same distribution and can share the samples
        # TODO could be optimized even more by splitting aggregators with multiple dtypes and only sampling new aggregating col values samples once per
        # dtype (rest of schema is same within template); but maybe performance gain not worth the added complexity
        aggregator_hash = hashlib.sha256(str.encode(str(operator.sql_allowed_types))).hexdigest()
        # use cached samples for variable assignments if possible for efficiency and performance comparability between operators
        if (cached_assignments := var_assignment_cache.get(aggregator_hash)) is None:
            var_assignments = template_obj.find_all_possible_assignments(sql_template, table)
            var_assignment_cache[aggregator_hash] = var_assignments
        else:
            var_assignments = cached_assignments

        # if condition is true create a template for the count operator additionally to the actual aggregator in the first iteration
        # a) for metadata statistics (field aggregation_num_rows of the dataset) if do_count_augmentation is True or
        # b) for explicit count questions in the dataset if self._explicit_count_definition is not None
        if (do_count_augmentation or template_obj._explicit_count_definition is not None) and op_idx == 0:
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
            if template_obj._explicit_count_definition is not None:
                compiled_count_nl_questions = [template_obj._nl_template.format(**dict(assignment,
                                                                                op=template_obj._explicit_count_definition.text,
                                                                                )
                                                                        )
                                                for assignment in count_var_assignments
                                                ]
                if create_alternatives:
                    compiled_count_alternatives = [template_obj._nl_template.format(**dict(assignment, op=alt))
                                                    for assignment in count_var_assignments
                                                    for alt in template_obj._explicit_count_definition.text_alternatives
                                                    ]
                    compiled_count_nl_questions += compiled_count_alternatives
                    compiled_count_sql_statements += (
                        len(compiled_count_alternatives)
                        * compiled_count_sql_statements
                    )

                count_aggregation_column_assignments = [template_obj.extract_aggregation_column(assignment) for assignment in count_var_assignments]
                condition_variables = template_obj.extract_condition_variables()
                count_condition_assignments = [template_obj.extract_condition_assignments(condition_variables, assignment) for assignment in count_var_assignments]

                count_questions = [
                    TableQuestion(
                        nl, table, sql, 'count',
                        aggregation_column=agg_col,
                        condition_assignments=condition_assign,
                        _template_hash=template_obj._template_hash,
                        _count_hash=count_config,
                        ).to_state_dict()
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
                count_questions = [TableQuestion('', table, sql, 'count', _template_hash=template_obj._template_hash, _count_hash=count_config)
                                    for sql, count_config in zip(compiled_count_sql_statements, count_configurations)
                                    ]
                generated_count_questions.extend(count_questions)

        # after potentially having initiallized the count aggregator and having computed the count_hash for all questions
        # fill the template slots for questions with the actual aggregator of the iteration
        compiled_sql_statements = [sql_template.format(**assignment)
                                    for assignment in var_assignments
                                    ]
        compiled_nl_questions = [template_obj._nl_template.format(**dict(assignment, op=operator.text))
                                    for assignment in var_assignments
                                    ]
        if create_alternatives:
            compiled_alternatives = [template_obj._nl_template.format(**dict(assignment, op=alt))
                                        for assignment in var_assignments
                                        for alt in operator.text_alternatives
                                        ]
            compiled_nl_questions += compiled_alternatives
            # TODO answer hashing in compute answer for efficiency
            compiled_sql_statements += (
                len(compiled_alternatives)
                * compiled_sql_statements
            )

        aggregation_column_assignments = [template_obj.extract_aggregation_column(assignment) for assignment in var_assignments]
        condition_variables = template_obj.extract_condition_variables()
        condition_assignments = [template_obj.extract_condition_assignments(condition_variables, assignment) for assignment in var_assignments]

        questions = [TableQuestion(nl, table, sql, operator.sql,
                                    aggregation_column=agg_col,
                                    condition_assignments=condition_cols,
                                    _template_hash=template_obj._template_hash,
                                    _count_hash=count_config,
                                    ).to_state_dict()
                        for nl, sql, agg_col, condition_cols, count_config in zip(
                            compiled_nl_questions,
                            compiled_sql_statements,
                            aggregation_column_assignments,
                            condition_assignments,
                            all_question_count_hashes or [None] * len(compiled_sql_statements),
                            )
                        ]
        generated_questions.extend(questions)

    return {'questions': generated_questions, 'count_questions': generated_count_questions}


def create_table_batch_questions(sample,  # sample of dataset if used with datasets map function, otherwise this is ignored
                                 template_obj: Optional[QuestionTemplate] = None,
                                 table: Optional[Table] = None,
                                 create_alternatives: bool = False,
                                 do_count_augmentation:bool = True,
                                 memory_mapped: bool = True,
                                 ) -> Union[Tuple[List[TableQuestion], List[TableQuestion]], Dict[str, List[TableQuestion]]]:
    if memory_mapped:  # either provide template_obj and table as arguments or load them from sample
        if template_obj is None:
            template_obj = QuestionTemplate.from_state_dict(sample['template'])
        if table is None:
            table = Table.from_state_dict(sample['table'])
    
    generated_questions = []
    generated_count_questions = []
    used_datatypes = [template_obj._schema['variables'][variable]['allowed_dtypes']
                        for variable in template_obj._schema['variables'].keys()
                        if template_obj._schema['variables'][variable]['type'] == 'value'
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
    for op_idx, operator in enumerate(template_obj._operators):
        # create Template object and generate template string for current operator (aggregator)
        sql_template_obj = SQLTemplate(
            SQLOperatorTemplate(
                operator.sql,
                template_obj.main_expression,
                brackets=None,
                over_clause=SQLOverClauseTemplate(
                    operator.sql_partition_cols,
                    operator.sql_order_cols,
                    operator.sql_frame_size
                ) if operator.sql_over_clause else None
            ),
            conditions=template_obj.conditions
        )
        sql_template = sql_template_obj.generate()

        # aggregator configurations with same allowed dtypes draw samples from the same distribution and can share the samples
        # TODO could be optimized even more by splitting aggregators with multiple dtypes and only sampling new aggregating col values samples once per
        # dtype (rest of schema is same within template); but maybe performance gain not worth the added complexity
        aggregator_hash = hashlib.sha256(str.encode(str(operator.sql_allowed_types))).hexdigest()
        # use cached samples for variable assignments if possible for efficiency and performance comparability between operators
        if (cached_assignments := var_assignment_cache.get(aggregator_hash)) is None:
            var_assignments = template_obj.find_all_possible_assignments(sql_template, table)
            var_assignment_cache[aggregator_hash] = var_assignments
        else:
            var_assignments = cached_assignments

        # if condition is true create a template for the count operator additionally to the actual aggregator in the first iteration
        # a) for metadata statistics (field aggregation_num_rows of the dataset) if do_count_augmentation is True or
        # b) for explicit count questions in the dataset if self._explicit_count_definition is not None
        if (do_count_augmentation or template_obj._explicit_count_definition is not None) and op_idx == 0:
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
            if template_obj._explicit_count_definition is not None:
                compiled_count_nl_questions = [template_obj._nl_template.format(**dict(assignment,
                                                                                op=template_obj._explicit_count_definition.text,
                                                                                )
                                                                        )
                                                for assignment in count_var_assignments
                                                ]
                if create_alternatives:
                    compiled_count_alternatives = [template_obj._nl_template.format(**dict(assignment, op=alt))
                                                   for assignment in count_var_assignments
                                                   for alt in template_obj._explicit_count_definition.text_alternatives
                                                   ]
                    compiled_count_nl_questions += compiled_count_alternatives
                    compiled_count_sql_statements += (
                        len(compiled_count_alternatives)
                        * compiled_count_sql_statements
                    )

                count_aggregation_column_assignments = [template_obj.extract_aggregation_column(assignment) for assignment in count_var_assignments]
                condition_variables = template_obj.extract_condition_variables()
                count_condition_assignments = [template_obj.extract_condition_assignments(condition_variables, assignment) for assignment in count_var_assignments]

                count_questions = [
                    TableQuestion(
                        nl, table, sql, 'count',
                        aggregation_column=agg_col,
                        condition_assignments=condition_assign,
                        _template_hash=template_obj._template_hash,
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
            else:
                count_questions = [TableQuestion('', table, sql, 'count', _template_hash=template_obj._template_hash, _count_hash=count_config)
                                    for sql, count_config in zip(compiled_count_sql_statements, count_configurations)
                                    ]
            # prepare for serialization if memory mapped
            if memory_mapped:
                count_questions = [q.to_state_dict() for q in count_questions]
            generated_count_questions.extend(count_questions)

        # after potentially having initiallized the count aggregator and having computed the count_hash for all questions
        # fill the template slots for questions with the actual aggregator of the iteration
        compiled_sql_statements = [sql_template.format(**assignment)
                                   for assignment in var_assignments
                                   ]
        compiled_nl_questions = [template_obj._nl_template.format(**dict(assignment, op=operator.text))
                                 for assignment in var_assignments
                                 ]
        if create_alternatives:
            compiled_alternatives = [template_obj._nl_template.format(**dict(assignment, op=alt))
                                     for assignment in var_assignments
                                     for alt in operator.text_alternatives
                                     ]
            compiled_nl_questions += compiled_alternatives
            # TODO answer hashing in compute answer for efficiency
            compiled_sql_statements += (
                len(compiled_alternatives)
                * compiled_sql_statements
            )

        aggregation_column_assignments = [template_obj.extract_aggregation_column(assignment) for assignment in var_assignments]
        condition_variables = template_obj.extract_condition_variables()
        condition_assignments = [template_obj.extract_condition_assignments(condition_variables, assignment) for assignment in var_assignments]

        questions = [TableQuestion(nl, table, sql, operator.sql,
                                    aggregation_column=agg_col,
                                    condition_assignments=condition_cols,
                                    _template_hash=template_obj._template_hash,
                                    _count_hash=count_config,
                                    ).to_state_dict()
                        for nl, sql, agg_col, condition_cols, count_config in zip(
                            compiled_nl_questions,
                            compiled_sql_statements,
                            aggregation_column_assignments,
                            condition_assignments,
                            all_question_count_hashes or [None] * len(compiled_sql_statements),
                            )
                        ]
        # prepare for serialization if memory mapped
        if memory_mapped:
            questions = [q.to_state_dict() for q in questions]
        generated_questions.extend(questions)
    
    if memory_mapped:
        return {'questions': generated_questions, 'count_questions': generated_count_questions}
    return generated_questions, generated_count_questions


def flatten_hierarchical_fields(sample,
                                old_field_names: Union[str, List[str]] = ['questions', 'count_questions'],
                                new_field_names: Optional[Union[str, List[str]]] = None,
                                reduce_field: str = 'table',  # hierarchy level that will be flattened
                                ):
    # always wrap field names in lists
    if isinstance(old_field_names, str):
        old_field_names = [old_field_names]
    if new_field_names is not None:
        if isinstance(new_field_names, str):
            new_field_names = [new_field_names]
        if len(new_field_names) != len(old_field_names):
            raise ValueError(f"old_field_names and new_field_names must have the same lengths but have {len(old_field_names)} and {len(new_field_names)} respectively!")

    return {field: [item for item in sample[reduce_field][old_field_names[f]]]
            for f, field in enumerate(new_field_names or old_field_names)  # if new_field_names are Nine use old_field_names instead
            }


def flatten_table_batches(table_dataset: datasets.Dataset, field_name='questions'):
    batch_datasets = [
        datasets.Dataset.from_list(
            [question
             for question in table_batch[field_name]
             ]
            )
        for table_batch in table_dataset
        ]
    flattened_question_dataset = datasets.concatenate_datasets(batch_datasets)
    for dataset in batch_datasets:
        delete_dataset(dataset)
    return flattened_question_dataset


def deduplicate_field(sample,
                      field_names: Optional[Union[str, List[str]]] = None,
                      object_class: Optional[Type] = None,
                      ):
    if field_names is None:
        field_names = list(field_names.keys())
    if isinstance(field_names, str):
        field_names = [field_names]

    # if class is provided create object from the data
    if object_class is not None:
        sample = {field: object_class.from_state_dict(sample[field])
                  for field in field_names
                  }
    # remove duplicates (might change original order)  
    sample = {field: list(set(sample[field]))
              for field in field_names
              }
    # if object was use apply serialization before output
    if object_class is not None:
        sample = {field: object_class.to_state_dict(sample[field])
                  for field in field_names
                  }     

    return sample


def cached_answer_computation(sample, field_names: Union[str, List[str]] = 'questions', sanswer_cache: dict = {}):
    if isinstance(field_names, str):
        field_names = [field_names]
    # deserialize to TableQuestion object
    deserialized_questions = {field: [TableQuestion.from_state_dict(question)
                                      for question in sample[field]
                                      ]
                              for field in field_names
                              }


def add_from_cache(sample, field_name: str = 'answer', cache: dict = {}, key_field: str = 'count_hash'):
    return {field_name: cache[sample[key_field]]}
