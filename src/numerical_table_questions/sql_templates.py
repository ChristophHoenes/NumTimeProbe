from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional, TypeVar, Type, TypeAlias


OP = TypeVar('OP', bound='SQLOperatorTemplate')
E = TypeVar('E', bound='SQLColumnExpression')
T = TypeVar('T', bound='SQLTemplate')
ExpressionArgument: TypeAlias = Union[str, int, E]


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


def get_operator_by_name(name: str) -> SQLOperator:
    match name.lower():
        case 'min': return MIN
        case 'max': return MAX
        case 'avg': return AVG
        case 'sum': return SUM
        case 'count': return COUNT
        case 'noop' | '': return NOOP
        case _: raise ValueError(f"Operator {name} is unknown! Please use either of (min, max, avg, sum, count or noop) or register your new operator in sql_templates.")


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
    def __init__(self,
                 arguments: Union[str, Tuple[ExpressionArgument, ExpressionArgument]],
                 operand: Optional[str] = None
                 ):
        if isinstance(arguments, tuple):
            if len(arguments) > 2:
                raise ValueError(f"Expected at most two arguments but {len(arguments)} were provided!")
            if operand is None and len(arguments) > 1:
                warnings.warn(f"{len(arguments)} arguments but no operand were specified! Only the first argument will be considered.")
        self._allowed_operands = ('+', '-', '*', '/', None)
        if operand not in self._allowed_operands:
            raise ValueError(f"Only operands in {self._allowed_operands} allowed, but found '{operand}'!")
        # allow lazy conversion of single string argument to tuple
        if isinstance(arguments, str):
            arguments = (arguments,)
        # make sure arguments is either a single variable or a hard-coded column name
        for argument in arguments:
            if isinstance(argument, str):
                num_brace_open = argument.count('{')
                num_brace_close = argument.count('}')
                if num_brace_open == 0 and num_brace_close == 0:
                    warnings.warn(f"Added hard-coded column name '{arguments}' to SQLColumnExpression, this will result in limited reusability of the SQL template!")
                elif num_brace_open >= 2 or num_brace_close >= 2:
                    raise ValueError("Encountered multiple variables within a single argument! For nested expressions pass SQLColumnExpressions as arguments.")
        self.arguments = arguments
        self.operand = operand  # Maybe operand class that defines allowed types

    def _generate_argument(self, argument: ExpressionArgument) -> str:
        match argument:
            case str():
                return f'"{argument}"'
            case int():
                return str(argument)
            case SQLColumnExpression():
                return f'({argument.generate()})'
            case _:
                raise TypeError(f"Type {type(argument)} is not allowed for an argument! Valid types are str, int and SQLColumnExpression.")

    def generate(self) -> str:
        if self.operand is None or len(self.arguments) == 1:
            if self.operand is not None:
                warnings.warn(f"Found operator '{self.operand}' but only one argument! "
                              "Operator will be ignored. Please check your template for validity."
                              )
            return self._generate_argument(self.arguments[0])
        return f"{self._generate_argument(self.arguments[0])} {self.operand} {self._generate_argument(self.arguments[1])}"

    def to_state_dict(self) -> dict:
        return {
            'argument0': self.arguments[0] if not isinstance(self.arguments[0], SQLColumnExpression) else None,
            'argument0_recursion': self.arguments[0].to_state_dict() if isinstance(self.arguments[0], SQLColumnExpression) else None,
            'argument1': self.arguments[1] if len(self.arguments) > 1 and not isinstance(self.arguments[1], SQLColumnExpression) else None,
            'argument1_recursion': self.arguments[1].to_state_dict() if len(self.arguments) > 1 and isinstance(self.arguments[1], SQLColumnExpression) else None,
            'operand': self.operand,
        }

    @classmethod
    def from_state_dict(cls, state_dict) -> SQLColumnExpression:
        instance = cls.__new__(cls)
        # only one argument
        if state_dict['argument1_recursion'] is None and state_dict['argument1'] is None:
            instance.arguments = (
                SQLColumnExpression.from_state_dict(state_dict['argument0_recursion']) if isinstance(state_dict['argument0_recursion'], dict) else state_dict['argument0'],
                )
        else:  # two arguments
            instance.arguments = (
                SQLColumnExpression.from_state_dict(state_dict['argument0_recursion']) if isinstance(state_dict['argument0_recursion'], dict) else state_dict['argument0'],
                SQLColumnExpression.from_state_dict(state_dict['argument1_recursion']) if isinstance(state_dict['argument1_recursion'], dict) else state_dict['argument1']
                )
        instance.operand = state_dict['operand']
        return instance


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
        # like wrapping single column name in double quotes is handled in condition_string property
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

    @property
    def condition_string(self) -> str:
        if isinstance(self.condition_column, SQLColumnExpression):
            return self.condition_column.generate()
        else:
            # TODO make sure in table processing that " does not occur in any column name
            if self.condition_column.startswith('"') and self.condition_column.endswith('"'):
                return self.condition_column
            else:
                return f'"{self.condition_column}"'

    @property
    def value_string(self) -> str:
        if isinstance(self.value, SQLColumnExpression):
            return self.value.generate()
        else:
            # two single quotes as fallback if value is empty (e.g empty string)
            return (self.value or r"''")

    def generate(self):
        return f"\n\tAND {self.condition_string} {self.comparator} {self.value_string}"


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
    # search for template variables and return unique variable names found while preserving order -> use dict instead of set
    return list({elem.strip('{}'): None for elem in re.findall(regex_pattern, template)}.keys())
