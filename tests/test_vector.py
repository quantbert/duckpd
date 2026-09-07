"""Typed vector retrieval behavior and execution-boundary coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import duckpd
from duckpd._logical import FilterPlan, Nullability, VectorSearchPlan
from duckpd.errors import MaterializationError, UnsupportedOperationError


def _table_frame() -> tuple[duckpd.Session, duckpd.DataFrame]:
    session = duckpd.connect()
    session._connection.execute(
        "CREATE TABLE documents(id INTEGER, kind VARCHAR, embedding FLOAT[3])"
    )
    session._connection.execute(
        """
        INSERT INTO documents VALUES
            (1, 'future', [1.0, 0.0, 0.0]),
            (2, 'eligible', [0.8, 0.2, 0.0]),
            (3, 'eligible', [0.0, 1.0, 0.0]),
            (4, 'eligible', [-1.0, 0.0, 0.0])
        """
    )
    return session, session.table("documents")


def _distance(left: tuple[float, ...], right: tuple[float, ...], metric: str) -> float:
    if metric == "l2":
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    if metric == "inner_product":
        return -dot
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return 1.0 - dot / (left_norm * right_norm)


@pytest.mark.parametrize("metric", ["cosine", "l2", "inner_product"])
def test_exact_search_matches_independent_brute_force(metric: str) -> None:
    session, frame = _table_frame()
    query = (1.0, 0.0, 0.0)
    execution_count = session.execution_count

    result = frame.vector.search(
        query,
        column="embedding",
        metric=metric,  # type: ignore[arg-type]
        k=4,
        tie_breaker="id",
    )

    assert session.execution_count == execution_count
    collected = result.collect()
    vectors = {
        1: (1.0, 0.0, 0.0),
        2: (0.8, 0.2, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (-1.0, 0.0, 0.0),
    }
    expected_ids = sorted(vectors, key=lambda key: (_distance(vectors[key], query, metric), key))
    assert collected["id"].tolist() == expected_ids
    assert collected["_distance"].tolist() == pytest.approx(
        [_distance(vectors[key], query, metric) for key in expected_ids],
        abs=1e-6,
    )
    assert session.execution_count == execution_count + 1


def test_exact_search_handles_empty_inputs_and_deterministic_ties() -> None:
    session = duckpd.connect()
    session._connection.execute("CREATE TABLE empty_vectors(id INTEGER, embedding FLOAT[3])")
    empty = session.table("empty_vectors").vector.search(
        [1, 0, 0], column="embedding", metric="l2", k=10, tie_breaker="id"
    )
    assert empty.collect().empty

    session._connection.execute(
        """
        INSERT INTO empty_vectors VALUES
            (2, [0.0, 1.0, 0.0]),
            (1, [0.0, -1.0, 0.0])
        """
    )
    tied = session.table("empty_vectors").vector.search(
        [1, 0, 0], column="embedding", metric="l2", k=10, tie_breaker="id"
    )
    assert tied.collect()["id"].tolist() == [1, 2]


def test_distance_expression_is_lazy_typed_and_assignable() -> None:
    session, frame = _table_frame()

    scored = frame.assign(score=frame["embedding"].vector.distance([1, 0, 0], metric="l2"))

    assert session.execution_count == 0
    assert scored._plan.metadata.visible_columns[-1].duckdb_type == "FLOAT"
    assert scored._plan.metadata.visible_columns[-1].nullable is Nullability.NON_NULL
    assert scored.collect()["score"].tolist() == pytest.approx(
        [0.0, math.sqrt(0.08), math.sqrt(2.0), 2.0], abs=1e-6
    )


def test_filter_before_and_after_search_remain_distinct() -> None:
    _, frame = _table_frame()
    eligible = frame[frame["kind"] == "eligible"]

    before = eligible.vector.search([1, 0, 0], column="embedding", k=2, tie_breaker="id")
    searched = frame.vector.search([1, 0, 0], column="embedding", k=2, tie_breaker="id")
    after = searched[searched["kind"] == "eligible"]

    assert isinstance(before._plan, VectorSearchPlan)
    assert isinstance(before._plan.input, FilterPlan)
    assert isinstance(after._plan, FilterPlan)
    assert isinstance(after._plan.input, VectorSearchPlan)
    assert before.collect()["id"].tolist() == [2, 3]
    assert after.collect()["id"].tolist() == [2]


def test_search_metadata_and_observability_are_explicit() -> None:
    session, frame = _table_frame()
    result = frame.vector.search(
        [1, 0, 0], column="embedding", metric="cosine", k=2, tie_breaker="id"
    )

    assert result.columns == ("id", "kind", "embedding", "_distance")
    assert result.ordering == ("_distance", "id")
    assert result._plan.metadata.row_identity == frame._plan.metadata.row_identity
    assert result._plan.metadata.provenance.row_preserving is False
    data = json.loads(result.explain("json"))
    vector = data["execution_boundaries"]["vector_operations"][0]
    assert vector == {
        "operation": "search",
        "strategy": "exact",
        "metric": "cosine",
        "dimension": 3,
        "k": 2,
        "filter_placement": "none",
        "tie_breaker": "id",
        "index_name": None,
        "physical_index_use": False,
    }
    assert session.execution_count == 0
    profile = result.profile()
    assert profile.vector_operations[0]["metric"] == "cosine"
    assert profile.vector_operations[0]["strategy"] == "exact"


def test_vector_validation_fails_before_source_execution() -> None:
    session, frame = _table_frame()
    invalid_calls = (
        lambda: frame.vector.search([1, 0], column="embedding"),
        lambda: frame.vector.search("100", column="embedding"),
        lambda: frame.vector.search(1, column="embedding"),
        lambda: frame.vector.search([], column="embedding"),
        lambda: frame.vector.search([True, 0, 0], column="embedding"),
        lambda: frame.vector.search([1, float("nan"), 0], column="embedding"),
        lambda: frame.vector.search([0, 0, 0], column="embedding", metric="cosine"),
        lambda: frame.vector.search(
            [1, 0, 0],
            column="embedding",
            metric="unknown",  # pyright: ignore[reportArgumentType]
        ),
        lambda: frame.vector.search([1, 0, 0], column="embedding", k=0),
        lambda: frame.vector.search([1, 0, 0], column="embedding", distance_column="kind"),
        lambda: frame.vector.search([1, 0, 0], column="embedding", distance_column=""),
        lambda: frame.vector.search([1, 0, 0], column="missing"),
        lambda: frame.vector.search([1, 0, 0], column="id"),
        lambda: frame.vector.search([1, 0, 0], column="embedding", mode="approximate"),
        lambda: frame.vector.search(
            [1, 0, 0],
            column="embedding",
            mode="invalid",  # pyright: ignore[reportArgumentType]
        ),
        lambda: frame.vector.search(
            [1, 0, 0],
            column="embedding",
            mode="approximate",
            tie_breaker="id",
        ),
        lambda: frame.vector.search([1, 0, 0], column="embedding", tie_breaker="embedding"),
    )
    for call in invalid_calls:
        with pytest.raises((KeyError, TypeError, ValueError, UnsupportedOperationError)):
            call()
    assert session.execution_count == 0


def test_invalid_source_vectors_fail_inside_single_query() -> None:
    session = duckpd.connect()
    session._connection.execute("CREATE TABLE invalid_vectors(id INTEGER, v FLOAT[2])")
    session._connection.execute(
        "INSERT INTO invalid_vectors VALUES (1, [1, 0]), (2, NULL), (3, [1, 'NaN'::FLOAT])"
    )
    result = session.table("invalid_vectors").vector.search([1, 0], column="v")

    with pytest.raises(MaterializationError):
        result.collect()
    assert session.execution_count == 1


def test_parquet_list_vectors_are_runtime_dimension_checked(tmp_path: Path) -> None:
    table = pa.table(
        {
            "id": pa.array([1, 2]),
            "embedding": pa.array([[1.0, 0.0], [0.0, 1.0]], type=pa.list_(pa.float32())),
        }
    )
    path = tmp_path / "vectors.parquet"
    import pyarrow.parquet as pq

    pq.write_table(table, path)  # pyright: ignore[reportUnknownMemberType]
    frame = duckpd.read_parquet(path)

    assert frame.vector.search([1, 0], column="embedding", k=1).collect()["id"].tolist() == [1]
    with pytest.raises(MaterializationError):
        frame.vector.search([1, 0, 0], column="embedding", k=1).collect()


def test_exact_search_composes_with_join_groupby_arrow_and_sink(tmp_path: Path) -> None:
    session, frame = _table_frame()
    labels = session.from_pandas(pd.DataFrame({"id": [1, 2, 3, 4], "weight": [2, 3, 4, 5]}))
    result = frame.vector.search([1, 0, 0], column="embedding", k=3).merge(labels, on="id")

    assert result.to_arrow().num_rows == 3
    grouped = result.groupby("kind", as_index=False).agg(total=("weight", "sum"))
    assert grouped.collect()["total"].sum() == 9
    output = tmp_path / "matches.parquet"
    result.write_parquet(output)
    assert pd.read_parquet(output).shape[0] == 3


def test_hostile_labels_compile_through_typed_expressions() -> None:
    session = duckpd.connect()
    session._connection.execute(
        'CREATE TABLE hostile("id""; DROP TABLE hostile; --" INTEGER, "vec""tor" DOUBLE[2])'
    )
    session._connection.execute("INSERT INTO hostile VALUES (1, [1.0, 0.0])")
    frame = session.table("hostile")

    result = frame.vector.search(
        [1, 0],
        column='vec"tor',
        distance_column='distance"value',
        tie_breaker='id"; DROP TABLE hostile; --',
    ).collect()

    assert result['distance"value'].tolist() == [0.0]
    assert session._connection.execute("SELECT count(*) FROM hostile").fetchone() == (1,)


def test_vector_index_lifecycle_and_verified_approximate_plan() -> None:
    session, frame = _table_frame()
    try:
        info = session.create_vector_index(
            table="documents",
            column="embedding",
            name="documents_embedding_hnsw",
            metric="cosine",
        )
    except UnsupportedOperationError as error:
        pytest.skip(f"DuckDB vss extension unavailable: {error}")

    assert info.persistent is False
    assert info.memory_limit_applies is False
    assert session.inspect_vector_indexes() == (info,)
    result = frame.vector.search(
        [1, 0, 0], column="embedding", metric="cosine", k=2, mode="approximate"
    )
    physical = result.explain("physical")
    assert "HNSW_INDEX_SCAN" in physical
    assert info.name in physical
    assert result.collect()["id"].tolist() == [1, 2]
    session.drop_vector_index(info.name)
    assert session.inspect_vector_indexes() == ()
