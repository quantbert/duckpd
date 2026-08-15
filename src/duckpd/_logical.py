"""Typed immutable logical state used by the public API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias
from uuid import UUID, uuid4

from duckpd._typing import ScalarValue


@dataclass(frozen=True)
class ColumnId:
    """Identity of a physical column independent of its displayed label."""

    value: UUID

    @classmethod
    def create(cls) -> ColumnId:
        return cls(uuid4())


@dataclass(frozen=True)
class Column:
    """Column metadata available without executing row-producing queries."""

    id: ColumnId
    label: str
    duckdb_type: str
    hidden: bool = False


@dataclass(frozen=True)
class PandasSource:
    """A pandas object retained by its owning session."""

    key: str


@dataclass(frozen=True)
class ArrowSource:
    """An Arrow object retained by its owning session."""

    key: str


@dataclass(frozen=True)
class ParquetSource:
    """One or more Parquet files scanned by DuckDB."""

    paths: tuple[str, ...]
    hive_partitioning: bool = False
    union_by_name: bool = False


@dataclass(frozen=True)
class CsvSource:
    """One or more CSV files scanned by DuckDB."""

    paths: tuple[str, ...]
    header: bool = True
    delimiter: str = ","
    auto_detect: bool = True


@dataclass(frozen=True)
class TableSource:
    """A table in the owning DuckDB connection."""

    name: str


@dataclass(frozen=True)
class SqlSource:
    """A validated read-only SQL query."""

    query: str


Source = (
    ArrowSource | CsvSource | PandasSource | ParquetSource | SqlSource | TableSource
)


class BinaryOperator(Enum):
    """Binary operations supported by the first expression compiler."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    TRUE_DIVIDE = "true_divide"
    MODULO = "modulo"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    LESS_THAN = "less_than"
    LESS_EQUAL = "less_equal"
    GREATER_THAN = "greater_than"
    GREATER_EQUAL = "greater_equal"
    AND = "and"
    OR = "or"


class UnaryOperator(Enum):
    """Unary operations supported by the first expression compiler."""

    INVERT = "invert"
    NEGATE = "negate"
    POSITIVE = "positive"


class AggregateOperator(Enum):
    """Global aggregate operations supported by the initial reduction API."""

    COUNT = "count"
    SIZE = "size"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    NUNIQUE = "nunique"
    ANY_VALUE = "any_value"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"
    QUANTILE = "quantile"
    ANY = "any"
    ALL = "all"


@dataclass(frozen=True)
class ColumnRef:
    """Reference a logical column by identity."""

    column_id: ColumnId


@dataclass(frozen=True)
class LiteralValue:
    """A scalar value represented safely by DuckDB."""

    value: ScalarValue


@dataclass(frozen=True)
class BinaryExpression:
    """Apply a binary operator to two expressions."""

    left: Expression
    operator: BinaryOperator
    right: Expression


@dataclass(frozen=True)
class UnaryExpression:
    """Apply a unary operator to an expression."""

    operator: UnaryOperator
    operand: Expression


@dataclass(frozen=True)
class FunctionCall:
    """Call a named scalar function with expressions."""

    name: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True)
class CastExpression:
    """Cast an expression to a target DuckDB type."""

    operand: Expression
    target_type: str


@dataclass(frozen=True)
class CaseWhen:
    """A SQL CASE WHEN cond THEN val ELSE default END expression."""

    condition: Expression
    value: Expression
    otherwise: Expression


@dataclass(frozen=True)
class WindowExpression:
    """An expression evaluated over a window specification."""

    function: str
    arguments: tuple[Expression, ...] = ()
    partition_by: tuple[Expression, ...] = ()
    order_by: tuple[SortKey, ...] = ()
    frame_spec: str | None = None


Expression: TypeAlias = (
    ColumnRef
    | LiteralValue
    | BinaryExpression
    | UnaryExpression
    | FunctionCall
    | CastExpression
    | CaseWhen
    | WindowExpression
)


@dataclass(frozen=True)
class NamedExpression:
    """An expression projected to a logical output column."""

    column: Column
    expression: Expression


@dataclass(frozen=True)
class AggregateExpression:
    """A pandas-aware aggregate over an optional input expression."""

    column: Column
    operator: AggregateOperator | None
    expression: Expression | None = None
    input_duckdb_type: str | None = None
    skipna: bool = True
    min_count: int = 0
    ddof: int = 1
    q: float = 0.5


class SortDirection(Enum):
    """Direction of a sort key."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class NullPlacement(Enum):
    """Placement of null values in a sort key."""

    FIRST = "first"
    LAST = "last"


@dataclass(frozen=True)
class SortKey:
    """A typed ordering expression."""

    expression: Expression
    direction: SortDirection
    null_placement: NullPlacement


class IndexUniqueness(Enum):
    """Known uniqueness state of an explicit index."""

    UNKNOWN = "unknown"
    UNIQUE = "unique"
    NON_UNIQUE = "non_unique"


@dataclass(frozen=True)
class IndexSpec:
    """Explicit index columns carried by a frame."""

    columns: tuple[ColumnId, ...] = ()
    drop: bool = True
    uniqueness: IndexUniqueness = IndexUniqueness.UNKNOWN


@dataclass(frozen=True)
class OrderColumn:
    """A guaranteed physical ordering key."""

    column_id: ColumnId
    direction: SortDirection
    null_placement: NullPlacement


@dataclass(frozen=True)
class OrderSpec:
    """Guaranteed output ordering, or no keys when order is unknown."""

    keys: tuple[OrderColumn, ...] = ()


@dataclass(frozen=True)
class FrameMetadata:
    """Schema, index, and ordering state for one logical plan."""

    columns: tuple[Column, ...]
    index: IndexSpec = IndexSpec()
    ordering: OrderSpec = OrderSpec()

    @property
    def visible_columns(self) -> tuple[Column, ...]:
        return tuple(column for column in self.columns if not column.hidden)


@dataclass(frozen=True)
class ExpressionMetadata:
    """Semantic properties used to validate and rewrite expressions."""

    is_elementwise: bool
    preserves_length: bool
    is_scalar_like: bool
    is_literal: bool
    has_window: bool = False
    order_dependency_count: int = 0


def expression_metadata(expression: Expression) -> ExpressionMetadata:
    """Infer metadata for the currently supported expression nodes."""
    if isinstance(expression, ColumnRef):
        return ExpressionMetadata(True, True, False, False)
    if isinstance(expression, LiteralValue):
        return ExpressionMetadata(True, True, True, True)
    if isinstance(expression, UnaryExpression):
        return expression_metadata(expression.operand)
    if isinstance(expression, CastExpression):
        return expression_metadata(expression.operand)
    if isinstance(expression, CaseWhen):
        cond_meta = expression_metadata(expression.condition)
        val_meta = expression_metadata(expression.value)
        other_meta = expression_metadata(expression.otherwise)
        return ExpressionMetadata(
            is_elementwise=(
                cond_meta.is_elementwise
                and val_meta.is_elementwise
                and other_meta.is_elementwise
            ),
            preserves_length=(
                cond_meta.preserves_length
                and val_meta.preserves_length
                and other_meta.preserves_length
            ),
            is_scalar_like=(
                cond_meta.is_scalar_like
                and val_meta.is_scalar_like
                and other_meta.is_scalar_like
            ),
            is_literal=(
                cond_meta.is_literal and val_meta.is_literal and other_meta.is_literal
            ),
            has_window=(
                cond_meta.has_window or val_meta.has_window or other_meta.has_window
            ),
            order_dependency_count=(
                cond_meta.order_dependency_count
                + val_meta.order_dependency_count
                + other_meta.order_dependency_count
            ),
        )
    if isinstance(expression, FunctionCall):
        arg_metas = [expression_metadata(arg) for arg in expression.arguments]
        return ExpressionMetadata(
            is_elementwise=all(m.is_elementwise for m in arg_metas),
            preserves_length=all(m.preserves_length for m in arg_metas),
            is_scalar_like=all(m.is_scalar_like for m in arg_metas),
            is_literal=all(m.is_literal for m in arg_metas),
            has_window=any(m.has_window for m in arg_metas),
            order_dependency_count=sum(m.order_dependency_count for m in arg_metas),
        )
    if isinstance(expression, WindowExpression):
        arg_metas = [expression_metadata(arg) for arg in expression.arguments]
        part_metas = [expression_metadata(p) for p in expression.partition_by]
        order_metas = [expression_metadata(k.expression) for k in expression.order_by]
        all_metas = arg_metas + part_metas + order_metas
        return ExpressionMetadata(
            is_elementwise=False,
            preserves_length=True,
            is_scalar_like=False,
            is_literal=False,
            has_window=True,
            order_dependency_count=(
                len(expression.order_by)
                + sum(m.order_dependency_count for m in all_metas)
            ),
        )

    left = expression_metadata(expression.left)
    right = expression_metadata(expression.right)
    return ExpressionMetadata(
        is_elementwise=left.is_elementwise and right.is_elementwise,
        preserves_length=left.preserves_length and right.preserves_length,
        is_scalar_like=left.is_scalar_like and right.is_scalar_like,
        is_literal=left.is_literal and right.is_literal,
        has_window=left.has_window or right.has_window,
        order_dependency_count=(
            left.order_dependency_count + right.order_dependency_count
        ),
    )


class LogicalPlanBase:
    """Shared metadata access for immutable plan nodes."""

    metadata: FrameMetadata

    @property
    def columns(self) -> tuple[Column, ...]:
        return self.metadata.columns


@dataclass(frozen=True)
class ScanPlan(LogicalPlanBase):
    """A lazy source scan."""

    source: Source
    metadata: FrameMetadata


@dataclass(frozen=True)
class FilterPlan(LogicalPlanBase):
    """Filter input rows by a boolean expression."""

    input: LogicalPlan
    predicate: Expression
    metadata: FrameMetadata


@dataclass(frozen=True)
class ProjectPlan(LogicalPlanBase):
    """Project expressions into a new schema."""

    input: LogicalPlan
    projections: tuple[NamedExpression, ...]
    metadata: FrameMetadata


@dataclass(frozen=True)
class SortPlan(LogicalPlanBase):
    """Establish row ordering."""

    input: LogicalPlan
    keys: tuple[SortKey, ...]
    metadata: FrameMetadata


@dataclass(frozen=True)
class LimitPlan(LogicalPlanBase):
    """Restrict the number of output rows."""

    input: LogicalPlan
    count: int
    offset: int
    metadata: FrameMetadata


@dataclass(frozen=True)
class AggregatePlan(LogicalPlanBase):
    """Aggregate an input frame with optional grouping keys."""

    input: LogicalPlan
    aggregates: tuple[AggregateExpression, ...]
    metadata: FrameMetadata
    keys: tuple[ColumnId, ...] = ()
    dropna: bool = True
    sort: bool = True


class JoinType(Enum):
    """Supported join types."""

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"


@dataclass(frozen=True)
class JoinPlan(LogicalPlanBase):
    """Join two logical plans."""

    left: LogicalPlan
    right: LogicalPlan
    how: JoinType
    left_keys: tuple[ColumnId, ...]
    right_keys: tuple[ColumnId, ...]
    metadata: FrameMetadata
    sort: bool = False


@dataclass(frozen=True)
class UnionPlan(LogicalPlanBase):
    """Concatenate multiple logical plans row-wise."""

    inputs: tuple[LogicalPlan, ...]
    metadata: FrameMetadata


LogicalPlan: TypeAlias = (
    ScanPlan
    | FilterPlan
    | ProjectPlan
    | SortPlan
    | LimitPlan
    | AggregatePlan
    | JoinPlan
    | UnionPlan
)
