from dataclasses import dataclass
from typing import Tuple, List, Union, Optional


@dataclass(frozen=True)
class SQLOperator:
    sql: str
    sql_allowed_types: Tuple[str, ...]
    text: str
    text_alternatives: Tuple[str, ...]
    sql_over_clause: bool = False
    sql_partition_cols: Tuple[str] = tuple()
    sql_order_cols: Tuple[str] = tuple()


# Pre-defined SQL window function operators
MIN = SQLOperator('min',
                  ('numeric',),
                  'minimum',
                  ('smallest value', 'lowest value', 'least value'),
                  False
                  )
MAX = SQLOperator('max',
                  ('numeric',),
                  'maximum',
                  ('largest value', 'highest value', 'biggest value', 'top value'),
                  False
                  )
AVG = SQLOperator('avg', ('numeric',), 'average', ('mean',), False)
SUM = SQLOperator('sum', ('numeric',), 'sum', ('total',), False)
NOOP = SQLOperator('', ('numeric',), 'value', tuple(), False)


class SQLColumnExpression:
    """Defines the structure of an SQL column expression.

        A column expression is either a single column name or another column expression
        that combines an binary (arithmetic) operand with another column name. 
        Recursively, long expressions with multiple operands can be constructed.

        Todos:
            - add check for allowed column types (all numeric due to arithmetics?)
            - add check for same column allowed constraints across chained expressions
            - revisit if multiple arguments or always exact two 
              (see ellipsis in type annotation)
    """
    def __init__(self, arguments: Tuple[Union[str, 'SQLColumnExpression'], ...],
                 operand: Optional[str] = None
                 ):
        self._allowed_operands = ('+', '-', '*', '/', None)
        assert operand in self._allowed_operands, \
            f"Only operands in {self._allowed_operands} allowed!"
        self.arguments = arguments
        self.operand = operand  # Maybe operand class that defines allowed types

    def generate(self):
        if self.operand is None:
            return f'"{self.arguments[0]}"'
        else:
            return (
                f'"{self.arguments[0]}"' if isinstance(self.arrguments[0], str)
                else f'({self.arrguments[0].generate()})'
                + self.operand
                + f'"{self.arguments[1]}"' if isinstance(self.arrguments[1], str)
                else f'({self.arrguments[1].generate()})'
            )


class SQLOverClauseTemplate:
    """Defines the structure of an SQL OVER clause."""

    def __init__(self, partition_columns: List[str],
                 order_columns: List[str],
                 frame_size: Tuple[Optional[int], Optional[int]] = (None, None)
                 ):
        assert len(partition_columns) > 0 or len(order_columns) > 0, \
            "Over Clause needs at least one column to perform  a partition and/or \
                order operation over!"
        self.partition_columns = partition_columns
        self.order_columns = order_columns
        self.frame_size = frame_size

    def generate(self):
        quoted_partition_columns = [f'"{col}"' for col in self.partition_columns]
        quoted_order_columns = [f'"{col}"' for col in self.order_columns]
        return (
            " OVER (" + f"\n\tPARTITION BY {','.join(quoted_partition_columns)}"
            if len(self.partition_columns) > 1 else "" +
            f"\n\tORDER BY {','.join(quoted_order_columns)}"
            if len(self.order_columns) > 1 else "" +
            f"\n\tROWS BETWEEN {self.frame_size[0] or 'UNBOUNDED'} PRECEEDING "
            f"AND {self.frame_size[1] or 'UNBOUNDED'} FOLLOWING\n)"
        )


class SQLOperatorTemplate:
    """Defines the structure of an SQL operator/aggregator (window) function).

        Todos:
            - add allowed_operators?
            - add allowed types (basic ones are all numerical but first_value can allow 
              text)
            - first_value must require over clause with ordering to be deterministic
    """

    def __init__(self,
                 operator_name: str,
                 expression: SQLColumnExpression,
                 brackets: Optional[str] = None,
                 over_clause: Optional[SQLOverClauseTemplate] = None
                 ):
        self.operator_name = operator_name
        self.brackets = brackets or "(" if self.operator_name != '' else ''
        self._allowed_brackets = {'': '', '(': ')'}
        self.expression = expression
        self.over_clause = over_clause
        self._is_integrity_valid = self.determine_integrity()

    def generate(self):
        if not self._is_integrity_valid:
            return None
        return (
            self.operator_name +
            self.brackets +
            self.expression.generate() +
            (')' if self.brackets == '(' else '') +
            (self.over_clause.generate() if self.over_clause is not None else '')
        )

    # TODO think about why not use assert instead?
    def determine_integrity(self, explain: bool = False
                            ) -> Union[bool, Tuple[bool, List[str]]]:
        """Executes integrity tests of the expression and returns results.

            If explain is set to True additional information on the reason
            for the test result of each component is provided.

            Args:
                explain (bool): Flag whether to provide explanation of
                the results or not

            Returns:
                bool: Whether the OperatorTemplate is compliant with all integrity
                      checks or not
                or tuple (if explain is True): Additional to the boolean a list of +
                                               explanations for the decision is provided
        """
        explanations = []
        if not isinstance(self.operator_name, str):
            explanations.append('operator_name must be a string')
            return False
        if not isinstance(self.expression.generate(), str) \
           or self.expression.generate() == '':
            explanations.append('expression must generate a non-empty string')
            return False
        if self.brackets not in self._allowed_brackets.keys():
            explanations.append(f'brackets must be in \
                                {list(self._allowed_brackets.keys())}'
                                )
            return False
        if self.over_clause is not None \
           and not isinstance(self.over_clause.generate(), str):
            explanations.append('over_clause must generate a string or it is invalid')
            return False
        if explain:
            return True if len(explanations) == 0 else False, explanations
        return True


class SQLConditionTemplate:
    """Defines the structure of an SQL WHERE filter condition."""

    def __init__(self,
                 condition: Union[str, SQLColumnExpression],
                 comparator: str,
                 value: Union[str, SQLColumnExpression]
                 ):
        # comparisons single(inter)/multiple(cross) column value =, <, > <=, >=
        # between range can be expressed with two conditions for now > x and < y
        # nested subquery can replace comparison value
        self._allowed_comparators = ('=', '<', '>', '<=', '>=', '!=')
        assert comparator in self._allowed_comparators, \
            "comparator must be in {self._allowed_comparators}!"
        self.comparator = comparator
        # must be of type SQLColumnExpression, str input just for ease of use -> cast
        self.condition_column = (SQLColumnExpression((condition,))
                                 if isinstance(condition, str)
                                 else condition
                                 )
        """
        # can be hard coded value but is cast to SQLColumnExpression if template string
        self.value = (SQLColumnExpression((value,))
                      if (isinstance(value, str)
                          and value.startswith('{')
                          and value.endswith('}'))
                      else value
                      )
        """
        self.value = value

    def generate(self):
        return (f"\n\tAND {self.condition_column.generate()} {self.comparator} " +
                (self.value.generate()
                 if isinstance(self.value, SQLColumnExpression)
                 else (self.value or r"''")   # two single quotes as fallback if empty
                 )
                )


def sql_template_from_components(operator: SQLOperatorTemplate,
                                 conditions: List[SQLConditionTemplate],
                                 table_specifier: str = 'df'
                                 ) -> str:
    """Generates final SQL template string from modular componets."""
    select_statement = "SELECT " + operator.generate() + " FROM " + table_specifier
    if len(conditions) > 0:
        select_statement += "\nWHERE true"
    for condition in conditions:
        select_statement += condition.generate()
    return select_statement
