import re
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from numerical_table_questions.sql_utils import execute_sql


class AnswerCoordinates:

    def __init__(self, column_id: int, row_ids: List[int], num_rows: int, num_cols: Optional[int] = None):
        self.column_id = column_id
        self.row_ids = sorted(row_ids)
        # initially num_rows is always positive (list of all the rows that are in the solution)
        self.num_rows = num_rows
        self.num_cols = num_cols
        # if the number of rows in the solution are more than the rows that are NOT in the solution
        # convert the row ids to negative row ids to save memory
        self.is_positive_row_ids = len(row_ids) <= (num_rows // 2)
        if self.is_positive_row_ids:
            self.row_ids = self._invert_row_ids()
            # calling _invert_row_ids always inverts the truth value of self.is_positive_row_ids
            # -> change back to original value
            self.is_positive_row_ids = not self.is_positive_row_ids

    def _process_column_mapping(self, col_mapping: Union[Dict[int, int], List[int]]) -> Dict[int, int]:
        if isinstance(col_mapping, list):
            if self.num_cols is None:
                raise TypeError("If num_cols is not provided (is None) col_mapping must be of type dict (not list) maping all column ids to a new id!")
            col_mapping = {original_id: new_id
                           for original_id, new_id in zip(range(len(col_mapping)), col_mapping)
                           }
        if len(col_mapping) != self.num_cols:
            raise ValueError(f"col_mapping must have a length equal to self.num_cols ({self.num_cols}) but was {len(col_mapping)} instead!")
        return col_mapping

    def permute_col(self, col_mapping: Union[Dict[int, int], List[int]]):
        col_mapping = self._process_column_mapping(col_mapping)
        new_col_id = col_mapping[self.column_id]
        if new_col_id is None:
            raise ValueError("Cannot delete the column the answer is based on!")
        self.column_id = col_mapping[self.column_id]

    def permute_rows(self, shuffle_mapping: Union[Dict[int, int], List[int]]):
        if isinstance(shuffle_mapping, list):
            shuffle_mapping = {original_id: new_id
                               for original_id, new_id in zip(range(len(shuffle_mapping)), shuffle_mapping)
                               }
        if len(shuffle_mapping) != self.num_rows:
            raise ValueError(f"shuffle_mapping must have a length equal to self.num_rows ({self.num_rows}) but was {len(shuffle_mapping)} instead!")
        new_row_ids = [shuffle_mapping[current_id] for current_id in self.row_ids]
        return new_row_ids

    def _invert_row_ids(self) -> List[int]:
        # invert truth value of positive vs. negative row_ids
        self.is_positive_row_ids = not self.is_positive_row_ids
        # sorted list of set difference between all row ids and current row ids
        return sorted(set(range(self.num_rows)) - set(self.row_ids))

    def generate(self) -> List[Tuple[int, int]]:
        # if row_ids are saved in negative form (ids that are not part of answer)
        # invert before generating the output
        temporary_invert = False
        if not self.is_positive_row_ids:
            temporary_invert = True
            self._invert_row_ids()

        norm_output_format = [(row_id, self.column_id) for row_id in self.row_ids]

        # invert the internal representation back to the original format
        if temporary_invert:
            self._invert_row_ids()
        return norm_output_format


class MultiColumnAnswerCoordinates(AnswerCoordinates):

    def __init__(self, column_ids: List[int], row_ids: List[int], num_rows: int, num_cols: Optional[int] = None):
        if len(column_ids) == 0:
            raise ValueError("column_ids contain at least one value!")
        elif len(column_ids) == 1:
            # if single column id is provided use standard AnswerCoordinates class
            super().__init__(column_ids[0], row_ids, num_rows, num_cols)
        else:
            # call super init with dummy value (integer to be consistent with signature typing) for column_id
            super().__init__(-100, row_ids, num_rows, num_cols)
            # replace column_id attribute with column_ids (List[int])
            self.column_ids = sorted(column_ids)
            delattr(self, 'column_id')

    def permute_col(self, col_mapping: Union[Dict[int, int], List[int]]):
        col_mapping = self._process_column_mapping(col_mapping)
        new_col_ids = [col_mapping[col] for col in self.column_ids]
        if None in new_col_ids:
            raise ValueError("Cannot delete a column the answer is based on! Make sure to provide a valid column mapping without None values!")
        self.column_ids = new_col_ids

    def generate(self) -> List[Tuple[int, int]]:
        column_coordinates = []
        for column_id in self.column_ids:
            # temporarily set self.column_id to be able to call super().generate() which uses self.column_id
            self.column_id = column_id
            column_coordinates.append(super().generate())
        # remove temporary self.column_id attribute (only if for loop was executed)
        if len(column_coordinates) > 0:
            delattr(self, 'column_id')
        return sum(column_coordinates, [])  # join all elements of the column_coordinates list


def compute_answer_coordinates(column_name: str, dataframe: pd.DataFrame, sql_query: str) -> AnswerCoordinates:
    if column_name == '_column_expression':
        select_without_condition = sql_query.split('from df')[0]  # assumes 'from df where' does not occur in aggregator or column name
        # each column name that occurs in the select statement is added to the answer column ids
        column_id = [c_id for c_id, col in enumerate(dataframe.columns)
                     if col in select_without_condition
                     ]
    elif column_name in dataframe.columns:
        column_id = dataframe.columns.get_loc(column_name)
    else:
        raise KeyError(f"Column name {column_name} was not found in table columns {list(dataframe.columns)}!")
    where_clause = re.search(r'from df\s+where', sql_query.lower())  # assumes 'from df where' does not occur in aggregator or column name
    if where_clause is None:
        # if no where clause exists all rows are selected
        answer_row_idxs = range(dataframe.shape[0])
    else:
        # determine row idxs that satisfy the where clause by executing the sql query with just the row idxs as selected column
        where_start = where_clause.start() if where_clause is not None else 'from df\nwhere true'
        df_copy_with_row_idxs = dataframe.assign(__row_idx__=pd.Series(range(dataframe.shape[0])))
        answer_set_query = 'SELECT "__row_idx__" ' + sql_query[where_start:]
        answer_row_idxs = execute_sql(answer_set_query, df_copy_with_row_idxs)
    if isinstance(column_id, list):
        return MultiColumnAnswerCoordinates(column_id, list(answer_row_idxs), dataframe.shape[0], dataframe.shape[1])
    return AnswerCoordinates(column_id, list(answer_row_idxs), dataframe.shape[0], dataframe.shape[1])


# for datasets map function
def posthoc_answer_coordinates(datasets_example):
    answer_coordinates = compute_answer_coordinates(datasets_example['aggregation_column'], table_df, datasets_example['sql'])
    answer_coordinates.generate()
