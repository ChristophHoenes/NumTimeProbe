import copy
import json
import logging
import math
import warnings
import weakref
from pathlib import PurePath
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
from tqdm import tqdm

from numerical_table_questions.data_caching import caching, delete_dataset
from numerical_table_questions.data_synthesis.memmep_data_synth import (
    add_from_cache, create_table_batch_questions, deduplicate_field, flatten_table_batches
    )
from numerical_table_questions.data_synthesis.question import TableQuestion
from numerical_table_questions.data_synthesis.table import Table
from numerical_table_questions.data_synthesis.question_template import QuestionTemplate


log_file_init_path = str(PurePath(__file__).parent.parent.parent.parent / 'logging.ini')
logging.config.fileConfig(log_file_init_path, disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class WeakRefableDict(dict):
    pass


class TableQuestionDataSet:

    def __init__(self,
                 name: str,
                 description: Optional[str] = None,
                 question_templates: Optional[List[QuestionTemplate]] = None,
                 tables: Optional[List[Table]] = None,
                 compute_answers=True,
                 compute_alternatives=False,
                 compute_coordinates=False,
                 allow_multiple_answers=True,
                 memeory_mapped=True,
                 ) -> None:
        self.name = name
        self.description = description
        self._question_templates = question_templates
        if isinstance(tables, datasets.Dataset):
            self._tables = tables
        else:
            self._tables = {table._table_id: table for table in (tables or [])}
        self._unanswerable_questions = []
        self._compute_coordinates = compute_coordinates
        self._questions = self._initialize_questions(question_templates,
                                                     tables,
                                                     compute_answers=compute_answers,
                                                     compute_alternatives=compute_alternatives,
                                                     memory_mapped=memeory_mapped,
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
    def tables(self) -> Union[List[Table]]:
        """Property containing a shallow copy of the list of lables in the dataset."""
        if isinstance(self._tables, datasets.Dataset):
            return self._tables
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
        # TODO update for datasets.Dataset serialization of tables and questions
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
        """Creates a huggingface datasets.Dataset from the (in memory) questions in this dataset."""
        logger.info('Grouping questions by table...')
        if isinstance(self._questions, datasets.Dataset):
            # join table and question datasets
            if not isinstance(self._tables, datasets.Dataset):
                table_dataset = datasets.Dataset.from_list([table.to_state_dict() for table in self._tables.values()])
            else:
                table_dataset = self._tables
            table_batches = []
            for table in table_dataset:
                questions = []
                question_lengths = []
                answers = []
                answer_lengths = []
                is_multy_row_answer = []
                sql = []
                aggregators = []
                aggregation_columns = []
                #aggregation_column_types = []  # missing for datasets.Dataset serialization
                num_conditions = []
                aggregation_num_rows = []
                for question in self._questions:
                    if question['table_id'] == table['table_id']:
                        questions.append(question['nl_question'])
                        question_lengths.append(len(question['nl_question']))
                        answers.append(question['answer'])
                        answer_lengths.append(len(question['answer']))
                        is_multy_row_answer.append(question['multi_row_answer'])
                        sql.append(question['sql_query'])
                        aggregators.append(question['operator'])
                        aggregation_columns.append(question['aggregation_column'])
                        # aggregation_column_types.append(question[''])  # TODO make consistent somehow with non-memmapped version (e.g. add here or remove there)
                        num_conditions.append(len(question['condition_assignments']))
                        aggregation_num_rows.append(str(question['num_rows_aggregated_in_answer']))
                table_batch_dataset = datasets.Dataset.from_list(
                    [{'table': table,
                      'questions': questions,
                      'question_lengths': question_lengths,
                      'answers': answers,
                      'answer_lengths': answer_lengths,
                      'is_multy_row_answer': is_multy_row_answer,
                      'sql': sql,  # needed in lazy processing if answer coordinates should be computed on the fly
                      'aggregators': aggregators,
                      'aggregation_columns': aggregation_columns,
                      'aggregation_column_types': aggregation_column_types,
                      'num_conditions': num_conditions,
                      'aggregation_num_rows': aggregation_num_rows,
                      }
                     ]
                    )
                table_batches.append(table_batch_dataset)
            return datasets.concatenate_datasets(table_batches)

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
                                                                 'sql': [question._sql_query],
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
                questions_by_table[question._table._table_id]['sql'].append(question._sql_query)
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
        sql = []
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
            sql.append(content_dict['sql'])
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
            'sql': sql,  # needed in lazy processing if answer coordinates should be computed on the fly
            'aggregators': aggregators,
            'aggregation_columns': aggregation_columns,
            'aggregation_column_types': aggregation_column_types,
            'num_conditions': num_conditions,
            'aggregation_num_rows': aggregation_num_rows,
        })

    def to_json(self):
        return json.dumps(self, default=lambda x: x.__dict__, sort_keys=True, indent=4)

    def _initialize_questions(self,
                              question_templates: List[QuestionTemplate],  # TODO option for datasets.Dataset of QuestionTemplate state dicts or str path
                              tables: List[Table],  # TODO option for datasets.Dataset of Table state dicts or str path
                              compute_answers=True,
                              compute_alternatives=False,
                              do_count_augmentation=True,
                              memory_mapped=True,
                              ) -> Union[List[TableQuestion], datasets.Dataset]:
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
        
        if memory_mapped:
            # start a huggingface datasets.Dataset with only the template hash and the
            if isinstance(tables, list):
                """ use create_table_batch_questions without map but it's not really mempry mapped without datasets -> just make a Dataset from_list
                for question_template in question_templates:
                    for table in tables:
                        question_dict = create_table_batch_questions(
                            None,  # positional argument sample is only used when function is passed to datasets .map()
                            template_obj=question_template,
                            table=table,
                            create_alternatives=compute_alternatives,
                            do_count_augmentation=do_count_augmentation,
                            memory_mapped=memory_mapped,
                            )
                """
                table_dataset = datasets.Dataset.from_list([{'table': table.to_state_dict()} for table in tables])
            #else:  # uncomment and indent if multi-row-commented block above is used
            elif isinstance(tables, str):
                table_dataset = caching(tables)
            elif isinstance(tables, datasets.Dataset):
                table_dataset = tables
            else:
                raise TypeError(f"Argument tables is expected to be of type List[Table], str (path to dataset) or datasets.Dataset, but found {type(tables)}!")
            for question_template in question_templates:
                hierarcical_question_dataset = table_dataset.map(
                    create_table_batch_questions,
                    fn_kwargs=dict(
                        template_obj=question_template,
                        create_alternatives=compute_alternatives,
                        do_count_augmentation=do_count_augmentation,
                        memory_mapped=memory_mapped,
                        )
                    )
                """ need question dataset instead (not all questions in memory)
                flattened_question_dataset, _ = hierarcical_question_dataset.map(
                    flatten_hierarchical_fields,
                    fn_kwargs=dict(
                        old_field_names=['questions', 'count_questions'],
                        reduce_field='table',
                        remove_columns=['table', 'questions', 'count_questions'],
                        )
                    ), delete_dataset(hierarcical_question_dataset)
                """
                deduplicated_question_dataset, _ = hierarcical_question_dataset.map(
                    deduplicate_field,
                    fn_kwargs=dict(
                        field_names=['questions', 'count_questions'],
                        object_class=TableQuestion,
                        )
                    ), delete_dataset(hierarcical_question_dataset)
                flattened_count_questions = flatten_table_batches(deduplicated_question_dataset, field_name='count_questions')
                flattened_questions = flatten_table_batches(deduplicated_question_dataset, field_name='questions')
                """ not possible to pass growing cache as kwarg as they are individual copies per process
                count_result_cache = {}
                count_answer_dataset, _ = deduplicated_question_dataset.map(
                    cached_answer_computation,
                    fn_kwargs=dict(
                        answer_cache=count_result_cache,
                        )
                    ), delete_dataset(deduplicated_question_dataset)
                del count_result_cache  # release memory for answer cache
                """
        else:
            # generate questions from templates and keep count operator and other operators seperate for now (more flexibility of when to
            # explicitly use count as questions vs. when to only infer pre-aggregation row count as statistical property of the question)
            question_batches, count_question_batches = zip(
                *[question_template.create_questions(tables,
                                                     do_count_augmentation=do_count_augmentation,
                                                     )
                for question_template in question_templates]
                )

            flattened_questions = [question
                                   for question_template_batch in question_batches
                                   for question in question_template_batch
                                   ]
            flattened_count_questions = [question
                                         for question_template_batch in count_question_batches
                                         for question in question_template_batch
                                         ]
            # remove duplicates
            flattened_questions = list(set(flattened_questions))
            flattened_count_questions = list(set(flattened_count_questions))

        def _to_question(question: Union[TableQuestion, dict]) -> TableQuestion:
            # if memory mapped the dataset contains the state dicts rather than the TableQuestion objects -> convert
            if not isinstance(question, TableQuestion):
                question = TableQuestion.from_state_dict(question)
            return question

        def _add_cached_field_to_dataset(dataset: datasets.Dataset, field_name: str, cache: dict, key: str):
            return dataset.map(
                add_from_cache,
                fn_kwargs=dict(
                    field_name=field_name,
                    cache=cache,  # will be copied num_proc times
                    key_field=key,
                    )
                )

        # compute and cache the results of counting rows per condition assignment
        count_result_cache = {}
        if do_count_augmentation:
            logger.info('Computing pre-aggregation row counts of table questions...')
            for question in (progress_bar := tqdm(flattened_count_questions)):
                progress_bar.set_description("Computing pre-aggregation row counts")
                question = _to_question(question)  # ensure TableQuestion object
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
            if isinstance(flattened_count_questions, datasets.Dataset):
                flattened_count_questions = _add_cached_field_to_dataset(flattened_count_questions, 'answer', count_result_cache, 'count_hash')

        # compute answers to the questions (use cached answers for questions with equivalent SQL)
        answer_result_cache = {}
        if compute_answers:
            logger.info('Computing answers to table questions...')
            for question in (progress_bar := tqdm(flattened_questions)):
                progress_bar.set_description("Computing answers to table questions")
                question = _to_question(question)  # ensure TableQuestion object
                if (cached_answer := answer_result_cache.get(question._sql_query)) is None:
                    question.compute_answer(compute_coordinates=self._compute_coordinates)
                    # cache answers for questions with same sql
                    answer_result_cache[question._sql_query] = question._answer
                else:
                    # do not compute same sql query twice, but use cached answer
                    question._answer = cached_answer
            if isinstance(flattened_questions, datasets.Dataset):
                flattened_questions = _add_cached_field_to_dataset(flattened_questions, 'answer', answer_result_cache, 'sql_query')
            self._is_answers_computed = compute_answers

            # determine which questions with count aggregator should be explicit questions in the dataset
            # rather than just used for meta data and add them
            explicit_count_questions = []
            has_template_explicit_count_question = {template.template_hash: template._explicit_count_definition is not None
                                                    for template in question_templates}
            for question in flattened_count_questions:
                question = _to_question(question)  # ensure TableQuestion object
                if has_template_explicit_count_question[question.template_hash]:
                    explicit_count_questions.append(question)
                    if question._answer is None:
                        logger.debug("Did not find any pre-computed answer for count question. Computing it now...")
                        question.compute_answer(compute_coordinates=self._compute_coordinates)
            if isinstance(flattened_questions, datasets.Dataset):
                explicit_count_question_dataset = datasets.Dataset.from_list([question.to_state_dict()
                                                                              for question in explicit_count_questions
                                                                              ]
                                                                             )
                flattened_questions = datasets.concatenate_datasets([flattened_questions, explicit_count_question_dataset])
            else:  # list of TableQuestions
                flattened_questions.extend(explicit_count_questions)

        # add row counts before aggregation as statistical property (meta data) of the TableQuestion
        if do_count_augmentation:
            if isinstance(flattened_questions, datasets.Dataset):
                flattened_questions = _add_cached_field_to_dataset(flattened_questions, 'num_rows_aggregated_in_answer', count_result_cache, 'count_hash')
            else:
                for question in flattened_questions:
                    question = _to_question(question)  # ensure TableQuestion object
                    # condition_hash = hashlib.sha256(str(tuple(question.condition_assignments)).encode()).hexdigest()
                    condition_hash = question._count_hash
                    if count_result_cache.get(condition_hash) is None:
                        print(condition_hash)
                    question._num_rows_aggregated_in_answer = count_result_cache.get(condition_hash, 'No count result')
        return flattened_questions

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
        if isinstance(self._tables, datasets.Dataset):
            warnings.warn("self._tables is already serialized as huggingface datasets.Dataset, no need to pickle.")
            return
        # removes weakrefs
        self._questions_by_table_id = lambda: None
        for table in self._tables.values():
            table.prepare_for_pickle()
