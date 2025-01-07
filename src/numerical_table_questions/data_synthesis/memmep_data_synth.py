import hashlib
from typing import Dict, List, Optional, Tuple, Type, Union

import datasets

from numerical_table_questions.data_caching import delete_dataset
from numerical_table_questions.data_synthesis.question import TableQuestion
from numerical_table_questions.data_synthesis.question_template import QuestionTemplate
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.sql_templates import SQLTemplate, SQLOperatorTemplate, SQLOverClauseTemplate


def create_all_question_fields(sample,
                               template_obj: QuestionTemplate,
                               create_alternatives: bool = False,
                               do_count_augmentation: bool = True,
                               max_num_value_samples: int = 10,
                               max_value_length: Optional[int] = None,
                               max_questions_per_table: Optional[int] = None,
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
                          for dtype in var_dtypes
                          ]
                         )
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
            var_assignments = template_obj.find_all_possible_assignments(sql_template,
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
                compiled_count_nl_questions = [
                    template_obj._nl_template.format(**dict(assignment, op=template_obj._explicit_count_definition.text))
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
                count_questions = [TableQuestion('_IMPLICIT_COUNT_NL', table, sql, 'count',
                                                 aggregation_column='_IMPLICIT_COUNT_AGG_COL',
                                                 condition_assignments=[('_IMPLICIT_COUNT_CONDITION_COL', '_IMPLICIT_COUNT_CONDITION_VAL')],
                                                 _template_hash=template_obj._template_hash,
                                                 _count_hash=count_config,
                                                 )
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
                                 do_count_augmentation: bool = True,
                                 max_num_value_samples: int = 10,
                                 max_value_length: Optional[int] = None,
                                 max_questions_per_table: Optional[int] = None,
                                 memory_mapped: bool = True,
                                 ) -> Union[Tuple[List[TableQuestion], List[TableQuestion]], Dict[str, List[TableQuestion]]]:
    if memory_mapped:  # either provide template_obj and table as arguments or load them from sample
        if template_obj is None:
            print("template empty")
            # checks for field template, otherwise assumes each sample is a QuestionTemplate state dict
            template_obj = QuestionTemplate.from_state_dict(sample['template'])
            # bad workaround since question key is added to sample -> no hierarchy but mixed state_dict
            #template_obj = QuestionTemplate.from_state_dict(sample.get('template', sample))
        if table is None:
            # checks for field table, otherwise assumes each sample is a Table state dict
            table = Table.from_state_dict(sample['table'])
            # bad workaround since question key is added to sample -> no hierarchy but mixed state_dict
            #table = Table.from_state_dict(sample.get('table', sample))

    generated_questions = []
    generated_count_questions = []
    used_datatypes = [template_obj._schema['variables'][variable]['allowed_dtypes']
                      for variable in template_obj._schema['variables'].keys()
                      if template_obj._schema['variables'][variable]['type'] == 'value'
                      ]
    used_datatypes = set([dtype
                          for var_dtypes in used_datatypes
                          for dtype in var_dtypes
                          ]
                         )
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
            var_assignments = template_obj.find_all_possible_assignments(sql_template,
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
                compiled_count_nl_questions = [
                    template_obj._nl_template.format(**dict(assignment, op=template_obj._explicit_count_definition.text))
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
                count_questions = [TableQuestion('_IMPLICIT_COUNT_NL', table, sql, 'count',
                                                 aggregation_column='_IMPLICIT_COUNT_AGG_COL',
                                                 condition_assignments=[('_IMPLICIT_COUNT_CONDITION_COL', '_IMPLICIT_COUNT_CONDITION_VAL')],
                                                 _template_hash=template_obj._template_hash,
                                                 _count_hash=count_config,
                                                 )
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
        #print("Template: ", sql_template, "Assignmenrt:", var_assignments[0], "Compiled SQL:", compiled_sql_statements[0])
        #raise Exception("Debug break!")
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
                                   )
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
            field_types = {}
            for q in questions:
                for key, value in q.items():
                    if field_types.get(key) is None:
                        field_types[key] = [type(value)]
                    else:
                        field_types[key].append(type(value))
            for key in field_types.keys():
                field_types[key] = list(set(field_types[key]))
            #print(field_types)
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


def extract_field_datasets(hierarchical_dataset: datasets.Dataset, field_names: Union[str, List[str], Tuple[str, ...]] = 'questions') -> List[datasets.Dataset]:
    if isinstance(field_names, str):
        field_name = field_names  # single field name
        # here each element of table_batch[field_name] should be a dictionary (each element represents an object with fields by itself)
        field_datasets = [
            datasets.Dataset.from_list(
                [sample for sample in table_batch[field_name]]
                )
            for table_batch in hierarchical_dataset
            ]
    elif isinstance(field_names, (list, tuple)) and all([isinstance(field_name, str) for field_name in field_names]):
        # if field_names is an iterable each element of table_batch[field_names[i]] should be a list of atomic items
        # (each item represents a field of an object; all fields of field_names need to be aggregated into a dict representing the object)
        try:
            field_datasets = [
                datasets.Dataset.from_list(
                    [{field_name: table_batch[field_name][i]
                      for field_name in field_names
                      }
                     for i in range(len(table_batch[field_names[0]]))
                     ]
                    )
                for table_batch in hierarchical_dataset
                ]
        except IndexError as e:
            raise IndexError("An index error occured! Make sure all field_names reference lists of the same length.") from e
    else:
        raise TypeError(f"Argument field_names must be one of str, List[str] or Tuple[str] but found a different type ({type(field_names)})!")
    return field_datasets


def join_batch_datasets(batch_datasets: List[datasets.Dataset], delete_batches: bool = False) -> datasets.Dataset:
    joined_dataset = datasets.concatenate_datasets(batch_datasets)
    if delete_batches:
        for dataset in batch_datasets:
            delete_dataset(dataset)
    return joined_dataset


def flatten_table_batches(table_dataset: datasets.Dataset,
                          field_names: Union[str, List[str], Tuple[str, ...]] = 'questions',
                          delete_batches: bool = False,
                          ):
    """ Flattens a dataset fith table batches into a dataset of the contents at field_names.
        If field_names is a string every item in table_dataset[field_names] is assumed to be a dictionary representing an object.
        If field_names is a list or tuple of strings every item in table_dataset[field_names[i]] is assumed to be an atomic value of the field in field_names[i].
        The fields in field_names are then aggregated into a dictionary representing an object. Every object is a sample of the output dataset.
        Note that in this case table_dataset[field_names[i]] needs to be a list with the same length for all i.
    """
    batch_datasets = extract_field_datasets(table_dataset, field_names)
    return join_batch_datasets(batch_datasets, delete_batches=delete_batches)


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
        sample = {field: [object_class.from_state_dict(state_dict) for state_dict in sample[field]]
                  for field in field_names
                  }
    # remove duplicates (might change original order)
    sample = {field: list(set(sample[field]))
              for field in field_names
              }
    # if object was used apply serialization before output
    if object_class is not None:
        sample = {field: [object_class.to_state_dict(obj) for obj in sample[field]]
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


# wrapper for add_from_cache map function for easier use
def add_cached_field_to_dataset(dataset: datasets.Dataset,
                                field_name: str,
                                cache: dict,
                                key: str,
                                num_proc: Optional[int] = None,
                                delete_old_version_cache: bool = True,
                                ) -> datasets.Dataset:
    dataset, _ = dataset.map(
        add_from_cache,
        fn_kwargs=dict(
            field_name=field_name,
            cache=cache,  # will be copied num_proc times
            key_field=key,
            ),
        desc=f"Add field '{field_name}' with cache...",
        num_proc=num_proc,
        ), delete_dataset(dataset) if delete_old_version_cache else None
    return dataset


# designed to work together with data_utils.lazy_multi_processing_posthoc_order
def table_batches_add_pre_aggregation_row_counts(table_batch_data: Tuple[int, Tuple[datasets.Dataset, dict]]) -> dict:
    idx = table_batch_data[0]
    table_batch = table_batch_data[1][0]
    count_questions = table_batch_data[1][1]['count_questions']
    cache = {question['count_hash']: question['answer']
             for question in count_questions
             }
    updated_batch = add_cached_field_to_dataset(
        table_batch,
        'num_rows_aggregated_in_answer',
        cache,
        key='count_hash',
        num_proc=None,  # already within multiprocessing --> only same process per table_batch
        delete_old_version_cache=True,
        )
    return {'idx': idx, 'data': updated_batch}


# TODO move to data_utils
def post_hoc_delete_questions_from_hierarchical(sample,
                                                condition_fields: Union[str, List[str]] = 'answers',
                                                condition_fn=lambda x: x in ['', 'None', 'nan'],
                                                fields: List[str] = ['questions', 'answers', 'sql'],
                                                ):
    if isinstance(condition_fields, str):
        condition_fields = [condition_fields]  # always wrap in list
    condition_field_generator = zip(*[sample[condition_field] for condition_field in condition_fields])
    delete_indices = [i for i, fields in enumerate(condition_field_generator) if condition_fn(*fields)]
    for field in fields:
        for delete_idx in sorted(delete_indices, reverse=True):
            del sample[field][delete_idx]
    return {field: sample[field] for field in fields}
