from __future__ import annotations

import copy
import hashlib
import re
import weakref
from collections.abc import Iterable
from typing import List, Literal, Optional, Self, Tuple, Union

import numpy as np
import pandas as pd


class Table:

    def __init__(self, data_dict: dict,
                 name: Optional[str] = None,
                 source_name: Optional[str] = None,
                 source_split: Optional[str] = None) -> None:
        assert not (name is None and data_dict.get('name') is None), \
            "Providing a table name is mandatory!\
            If  name is not specified in data_dict it must be passed as an argument \
            explicitly."
        self._data_dict = data_dict
        # function instead of default value because uses weak reference after init
        # which requires calling (parentheses) for access
        self._dataframe = lambda: None
        self.table_name = name or self._data_dict['name']
        self._preprocess_column_names()  # fills empty column names
        self.column_names = tuple(
            self.deduplicate_column_names(self._data_dict['header'])
        )
        self._source = source_name
        self._source_split = source_split

        # data processing/cleaning and dtype inference
        self._preprocess_cells()   # handles trailing whitespaces and empty values
        # infer column types after whitespace removal
        self._inferred_column_types = [self.infer_column_type(col)
                                       for col in self.column_names
                                       ]
        self._make_true_numeric()  # removes commas (e.g 10,000.99 -> 10000.99)

        self._col2idx, self._idx2col = name_id_mapping(self.column_names, both_ways=True)
        sorted_col_names, arg_sort_col_idxs = alpha_numeric_sort(self.column_names)
        self._col_sort_idxs = arg_sort_col_idxs  # save sort idxs for later use
        # hash of table schema in terms of column names and types
        self._table_schema_id = hashlib.sha256(
            str.encode(
                ''.join(sorted_col_names) +
                ''.join(
                    alpha_numeric_sort(
                        self._inferred_column_types,
                        indices=arg_sort_col_idxs,
                        )[0]  # only hash sorted values not the sort indices
                    )
                )
            ).hexdigest()
        # sequence of sums of every numerical column (ordered alphabetically by column name) used as checksum for approximate identity
        self._num_sum_id = self._compute_num_sum_id()
        # only approximate identity for efficiency (not all values are checked)
        self._table_id = hashlib.sha256(str.encode(self._table_schema_id + self._num_sum_id + str(self.size))).hexdigest()

    @classmethod
    def from_state_dict(cls, state_dict) -> Self:
        """ Creates empty instance and loads the serialized values from the state_dict
            instead of recomputing them.
        """
        instance = cls.__new__(cls)
        instance._data_dict = state_dict['data_dict']
        instance._dataframe = lambda: None
        instance.table_name = state_dict['table_name']
        instance.column_names = state_dict['deduplicated_column_names']
        instance._source = state_dict['source_name']
        instance._source_split = state_dict['source_split']
        instance._inferred_column_types = state_dict['inferred_column_types']
        instance._col2idx, instance._idx2col = name_id_mapping(instance.column_names, both_ways=True)
        instance._table_id = state_dict['table_id']
        return instance

    @property
    def pandas_dataframe(self) -> Union[pd.DataFrame, weakref.ReferenceType[pd.DataFrame]]:
        """This property transforms the internal structure of the table data into a
            pandas DataFrame.
        """
        if self._dataframe() is None:
            df = pd.DataFrame.from_dict(
                {i: row
                 for i, row in enumerate(self._data_dict['rows'])
                 },
                orient='index',
                columns=self.column_names
            )
            self._dataframe = weakref.ref(df)
            return df
        return self._dataframe()

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

    def _compute_num_sum_id(self):
        sums_of_sorted_numerical_columns = [self.pandas_dataframe[col].sum()
                                            for col in alpha_numeric_sort(self.column_names, self._col_sort_idxs)[0]
                                            if self._inferred_column_types[self._col2idx[col]] == 'numeric'
                                            ]
        return ';'.join([str(col_sum) for col_sum in sums_of_sorted_numerical_columns])

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
            return list(self.pandas_dataframe.get(column_name, default=pd.Series()).unique())
        else:
            return list(self.pandas_dataframe.get(column_name, default=pd.Series()))

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
                          ) -> Literal['numeric', 'text', 'alphanumeric']:
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

        # determine row indices of the value samples
        if row_selection == 'first':
            sample_row_idxs = np.arange(num_test_rows)
        else:
            sample_row_idxs = np.random.choice(np.arange(len(self)),
                                               min(num_test_rows, len(self)),
                                               replace=False)
        # select sample cells from df
        df = self.pandas_dataframe
        sample_rows = df.iloc[sample_row_idxs, df.columns.get_loc(column_name)]

        # determine dtype of column based on samples
        def is_numeric(x, strict=True, number_regex=NUMBER_REGEX):
            if strict:  # true numeric (whole string must be a pure number)
                return re.fullmatch(number_regex, x) is not None
            else:
                return re.search(number_regex, x) is not None

        numeric_test_results = [is_numeric(row) for row in sample_rows]
        alphanumeric_test_results = [is_numeric(row, strict=False) for row in sample_rows]
        if all(numeric_test_results):
            return 'numeric'
        elif all(alphanumeric_test_results):
            return 'alphanumeric'
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
        while len(set([col.lower() for col in column_names])) != len(column_names):
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
                if col_name_counter.get(col_name.lower()) is None:
                    col_name_counter[col_name.lower()] = 1
                    new_col_names.append(col_name)
                else:
                    col_name_counter[col_name.lower()] += 1
                    new_col_names.append(
                        col_name
                        + f"{extension_string}"
                        + f"{col_name_counter[col_name.lower()] if use_numbering else ''}"
                    )
            column_names = new_col_names
            while_counter += 1
        return column_names

    def _make_true_numeric(self):
        num_col_ids = [idx for idx, typ in enumerate(self._inferred_column_types)
                       if typ == 'numeric']
        for r, row in enumerate(self._data_dict['rows']):
            for num_col in num_col_ids:
                self._data_dict['rows'][r][num_col] = row[num_col].replace(',', '')

    def sample_values(self, col_name):
        raise NotImplementedError

    def _preprocess_column_names(self):
        for c, column in enumerate(self._data_dict['header']):
            # empty column names are replaced with column_ + id
            self._data_dict['header'][c] = column or f'column_{c}'

    def _preprocess_cells(self):
        for r, row in enumerate(self._data_dict['rows']):
            for v, value in enumerate(row):
                # remove trailing whitespaces
                self._data_dict['rows'][r][v] = value.strip()
                if value == '':
                    # all empty values are marked as such via double single quotes
                    self._data_dict['rows'][r][v] = "''"

    def prepare_for_pickle(self):
        # weakref cannot be pickled, hence replace it with default value
        self._dataframe = lambda: None

    def to_state_dict(self):
        return {
            'table_id': self._table_id,
            'table_name': self.table_name,
            'source_name': self._source,
            'source_split': self._source_split,
            'data_dict': self._data_dict,
            'deduplicated_column_names': self.column_names,
            'inferred_column_types': self._inferred_column_types,
        }

    def __len__(self):
        return self.size[1]


# TODO maybe to utils
def name_id_mapping(names: List[str], both_ways: bool = False):
    name2id = {name: idx for idx, name in enumerate(names)}
    if both_ways:
        id2name = {idx: name for name, idx in name2id.items()}
        return name2id, id2name
    return name2id


def alpha_numeric_sort(items: Iterable, indices: Optional[Iterable[int]] = None) -> Tuple[Tuple[str], Tuple[int]]:
    if indices is not None:
        if len(items) != len(indices):
            raise ValueError(f"items and indices must have same size! But they have {len(items)} and {len(indices)} items, respectively.")
        return [items[idx] for idx in indices], indices
    sorted_list = sorted([(str(item), i) for i, item in enumerate(items)])
    sorted_items, arg_sort_idxs = zip(*sorted_list)
    return sorted_items, arg_sort_idxs
