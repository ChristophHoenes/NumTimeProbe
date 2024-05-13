import re
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional, TypeVar, Type


OP = TypeVar('OP', bound='SQLOperatorTemplate')
T = TypeVar('T', bound='SQLTemplate')

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
# Note that there is some special behaviour implemented for the count operator for efficiency reasons
# (e.g. count of rows will be the same regardless the column it is applied to -> only add once to dataset for every condition assignment)
COUNT = SQLOperator('count', ('numeric', 'text', 'alphanumeric'), 'count', ('number of rows',), False)
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
    def __init__(self, arguments: Union[str, Tuple[Union[str, 'SQLColumnExpression'], ...]],
                 operand: Optional[str] = None
                 ):
        self._allowed_operands = ('+', '-', '*', '/', None)
        assert operand in self._allowed_operands, \
            f"Only operands in {self._allowed_operands} allowed!"
        # allow lazy conversion of single string argument to tuple
        if isinstance(arguments, str):
            arguments = (arguments,)
        self.arguments = arguments
        self.operand = operand  # Maybe operand class that defines allowed types

    def generate(self):
        if self.operand is None:
            return f'"{self.arguments[0]}"'
        else:
            if isinstance(self.arguments[0], str):
                first_arg = f'"{self.arguments[0]}"'
            else:
                first_arg = f'({self.arguments[0].generate()})'
            if isinstance(self.arguments[1], str):
                second_arg = f'"{self.arguments[1]}"'
            else:
                second_arg = f'({self.arguments[1].generate()})'
        return f"{first_arg} {self.operand} {second_arg}"


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
                 over_clause: Optional[SQLOverClauseTemplate] = None,
                 ):
        self.operator_name = operator_name
        self.brackets = brackets or "(" if self.operator_name != '' else ''
        self._allowed_brackets = {'': '', '(': ')'}
        self.expression = expression
        self.over_clause = over_clause
        self._is_integrity_valid = self.determine_integrity()

    @classmethod
    def from_template(cls: Type[OP],
                      template: OP,
                      operator_name: str = None,
                      expression: SQLColumnExpression = None,
                      brackets: Optional[str] = None,
                      over_clause: Optional[SQLOverClauseTemplate] = None,
                      ) -> OP:
        return cls(operator_name=operator_name or template.operator_name,
                   expression=expression or template.expression,
                   brackets=brackets or template.brackets,
                   over_clause=over_clause or template.over_clause,
                   )

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
                 ) -> Union[str, Dict[str, str]]:
        # comparisons single(inter)/multiple(cross) column value =, <, > <=, >=
        # between range can be expressed with two conditions for now > x and < y
        # nested subquery can replace comparison value
        self._allowed_comparators = ('=', '<', '>', '<=', '>=', '!=')
        assert comparator in self._allowed_comparators, \
            "comparator must be in {self._allowed_comparators}!"
        self.comparator = comparator
        # correct processing based on datatype (str or SQLColumnExpression)
        # like wrapping single column name in double quotes is handled in generate
        self.condition_column = condition
        """
        # can be hard coded value but is cast to SQLColumnExpression if template string
        self.value = (SQLColumnExpression((value,))
                      if (isinstance(value, str)
                          and value.startswith('{')
                          and value.endswith('}'))
                      else value
                      )
        """
        if isinstance(value, str) and len(find_template_variables(value)) > 1:
            warnings.warn("Encountered more than one variable inside the value of the condition! "
                          "The first value variable is generated/sampled according to self.condition_column. "
                          "Make sure all additional variables occur in a previous condition or are of type column.")
        self.value = value

    def generate(self, in_parts=False):
        if isinstance(self.condition_column, SQLColumnExpression):
            condition_str = self.condition_column.generate()
        else:
            if self.condition_column.startswith('"') and self.condition_column.endswith('"'):
                condition_str = self.condition_column
            else:
                condition_str = f'"{self.condition_column}"'
        if isinstance(self.value, SQLColumnExpression):
            value_str = self.value.generate()
        else:
            # two single quotes as fallback if value is empty (e.g empty string)
            value_str = (self.value or r"''")
        if in_parts:
            return {'condition_column': condition_str, 'comparator': self.comparator, 'value': value_str}
        return f"\n\tAND {condition_str} {self.comparator} {value_str}"


class SQLTemplate:
    def __init__(self,
                 operator: SQLOperatorTemplate,
                 conditions: List[SQLConditionTemplate],
                 table_specifier: str = 'df',
                 ):
        self.operator = operator
        self.conditions = conditions
        self.table_specifier = table_specifier

    @classmethod
    def from_template(cls: Type[T],
                      template: T,
                      operator: SQLOperatorTemplate = None,
                      conditions: List[SQLConditionTemplate] = None,
                      table_specifier: str = 'df',
                      ) -> T:
        return cls(operator=operator or SQLOperatorTemplate.from_template(template.operator),
                   conditions=conditions or template.conditions,
                   table_specifier=table_specifier or template.table_specifier,
                   )

    def generate(self):
        """Generates final SQL template string from modular componets."""
        select_statement = "SELECT " + self.operator.generate() + " FROM " + self.table_specifier
        if len(self.conditions) > 0:
            select_statement += "\nWHERE true"
        for condition in self.conditions:
            select_statement += condition.generate()
        return select_statement


def find_template_variables(template: str) -> List[str]:
    regex_pattern = r'\{[^\{]+\}'  # any sequence of chars in braces (excluding opening brace char) where len > 0
    return [elem.strip('{}') for elem in re.findall(regex_pattern, template)]
