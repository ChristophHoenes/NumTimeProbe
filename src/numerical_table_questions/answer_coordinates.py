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

    def permute_col(self, col_mapping: Union[Dict[int, int], List[int]]):
        if isinstance(col_mapping, list):
            if self.num_cols is None:
                raise TypeError("If num_cols is not provided (is None) col_mapping must be of type dict (not list) maping all column ids to a new id!")
            col_mapping = {original_id: new_id
                           for original_id, new_id in zip(range(len(col_mapping)), col_mapping)
                           }
        if len(col_mapping) != self.num_cols:
            raise ValueError(f"col_mapping must have a length equal to self.num_cols ({self.num_cols}) but was {len(col_mapping)} instead!")
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

        norm_output_format = [(self.column_id, row_id) for row_id in self.row_ids]

        # invert the internal representation back to the original format
        if temporary_invert:
            self._invert_row_ids()
        return norm_output_format


def compute_answer_coordinates(column_name: str, dataframe: pd.DataFrame, query: str) -> AnswerCoordinates:
    if column_name in dataframe.columns:
        column_id = dataframe.columns.get_loc(column_name)
    else:
        raise KeyError(f"Column name {column_name} was not found in table columns {list(dataframe.columns)}!")
    where_start = re.search(r'from df\s+where', query.lower()).start()  # assumes 'from df where' does not occur in aggregator or column name
    df_copy_with_row_idxs = dataframe.assign(__row_idx__=pd.Series(range(dataframe.shape[0])))
    answer_set_query = 'SELECT "__row_idx__" ' + query[where_start:]
    answer_row_idxs = execute_sql(answer_set_query, df_copy_with_row_idxs)
    return AnswerCoordinates(column_id, list(answer_row_idxs), dataframe.shape[0], dataframe.shape[1])
