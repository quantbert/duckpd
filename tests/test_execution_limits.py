from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import duckpd


def test_resource_limits_and_spill_directory_execution() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_dir = Path(tmpdir) / "spill"
        spill_dir.mkdir(parents=True, exist_ok=True)

        with duckpd.connect(
            memory_limit="32MB",
            temp_directory=spill_dir,
            max_temp_directory_size="1GB",
            threads=2,
        ) as session:
            # Generate 250,000 rows
            frame = session.sql(
                """
                SELECT 
                    (i % 100)::INTEGER as group_key,
                    (i * 1.5)::DOUBLE as value,
                    (i % 2 = 0)::BOOLEAN as flag
                FROM range(250000) t(i)
                """
            )

            # Sort and write parquet directly
            out_parquet = Path(tmpdir) / "sorted.parquet"
            sorted_frame = frame.sort_values("value", ascending=False)
            sorted_frame.write_parquet(out_parquet)

            assert out_parquet.exists()
            assert out_parquet.stat().st_size > 0

            # GroupBy aggregation
            grouped = frame.groupby("group_key", as_index=False).agg(
                total_val=("value", "sum"),
                count_flag=("flag", "count"),
                size=("value", "size"),
            )
            res = grouped.collect()

            assert isinstance(res, pd.DataFrame)
            assert res.shape == (100, 4)
            assert session.execution_count == 2


def test_constrained_memory_execution_completes() -> None:
    """A generated workload completes with strict DuckDB resource settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        spill_dir = Path(tmpdir) / "spill_strict"
        spill_dir.mkdir(parents=True, exist_ok=True)

        with duckpd.connect(
            memory_limit="32MB",
            temp_directory=spill_dir,
            max_temp_directory_size="500MB",
            threads=1,
        ) as session:
            frame = session.sql(
                """
                SELECT
                    (i % 50)::BIGINT as grp,
                    (i * 3.14159)::DOUBLE as val1,
                    (i * 2.71828)::DOUBLE as val2,
                    ('prefix_' || (i % 1000)::VARCHAR) as label
                FROM range(500000) t(i)
                """
            )

            sorted_out = Path(tmpdir) / "stress_sorted.parquet"
            frame.sort_values(["val1", "val2"], ascending=[False, True]).write_parquet(
                sorted_out
            )
            assert sorted_out.exists()
            assert sorted_out.stat().st_size > 0

            reduced = (
                frame.groupby("grp", as_index=False)
                .agg(
                    s1=("val1", "sum"),
                    m2=("val2", "mean"),
                    cnt=("label", "count"),
                )
                .collect()
            )
            assert len(reduced) == 50


def test_constrained_memory_spill_and_isolated_rss() -> None:
    """Validate DuckDB spills to disk and child RSS is bounded under memory limits."""
    child_code = """
import json
import sys
import tempfile
from pathlib import Path

import duckpd
from benchmark.metrics import get_peak_rss_bytes
with tempfile.TemporaryDirectory() as tmpdir:
    spill_dir = Path(tmpdir) / "spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    with duckpd.connect(
        memory_limit="20MB",
        temp_directory=spill_dir,
        max_temp_directory_size="1GB",
        threads=1,
    ) as session:
        frame = session.sql(\"\"\"
            SELECT
                i as id,
                (i * 1.5)::DOUBLE as val,
                repeat('x', 100) as payload
            FROM range(500000) t(i)
        \"\"\")
        sorted_frame = frame.sort_values("val", ascending=False)
        agg = sorted_frame.groupby("id", as_index=False).agg(cnt=("val", "count"))
        prof = agg.profile()

        print(json.dumps({
            "peak_temp_dir_size": prof.peak_temp_dir_size,
            "peak_buffer_memory": prof.peak_buffer_memory,
            "child_rss_bytes": get_peak_rss_bytes(),
            "rows_returned": prof.rows_returned,
        }))
"""
    proc = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout.strip())
    assert result["rows_returned"] == 500000
    assert result["peak_temp_dir_size"] > 0, "DuckDB must spill to disk"
    assert result["peak_buffer_memory"] > 0
    if result["child_rss_bytes"] > 0:
        assert result["child_rss_bytes"] < 350 * 1024 * 1024


def test_explain_modes() -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))

    logical = frame.explain(mode="logical")
    assert "DuckPD logical plan:" in logical
    assert "DuckDB SQL:" not in logical

    sql = frame.explain(mode="sql")
    assert "DuckDB SQL:" in sql
    assert "DuckPD logical plan:" not in sql

    physical = frame.explain(mode="physical")
    assert "DuckDB physical plan:" in physical
    assert "DuckPD logical plan:" not in physical

    all_views = frame.explain(mode="all")
    assert "DuckPD logical plan:" in all_views
    assert "DuckDB SQL:" in all_views
    assert "DuckDB physical plan:" in all_views

    with pytest.raises(ValueError, match="Unknown explain mode"):
        frame.explain(mode="invalid")  # type: ignore[arg-type]


def test_explain_write(tmp_path: Path) -> None:
    frame = duckpd.from_pandas(pd.DataFrame({"x": [1, 2, 3]}))
    target = tmp_path / "out.parquet"

    info = frame.explain_write(target, compression="zstd")
    assert f"Write target: {target}" in info
    assert "Compression: zstd" in info
    assert "Output columns:" in info
    assert "DuckDB physical plan:" in info


def test_dataframe_profile() -> None:
    from duckpd import ProfileResult

    frame = duckpd.from_pandas(pd.DataFrame({"a": range(100), "b": range(100)}))
    filtered = frame[frame["a"] > 50].assign(c=lambda f: f["a"] * 2)

    prof = filtered.profile()
    assert isinstance(prof, ProfileResult)
    assert prof.latency > 0.0
    assert prof.rows_scanned == 100
    assert prof.rows_returned == 49
    assert prof.bytes_read >= 0
    assert prof.bytes_written >= 0
    assert prof.peak_buffer_memory >= 0
    assert prof.peak_temp_dir_size >= 0

    summary = prof.summary()
    assert "DuckPD Query Profile Summary" in summary
    assert "Execution Latency:" in summary
    assert "Rows Scanned:" in summary
    assert repr(prof) == summary
    assert str(prof) == summary

    data = prof.to_dict()
    assert isinstance(data, dict)
    assert "cumulative_rows_scanned" in data

    json_str = prof.to_json()
    assert isinstance(json_str, str)
    assert '"cumulative_rows_scanned": 100' in json_str
