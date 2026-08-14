from __future__ import annotations

import pytest

from demo.reduction_pipeline import main


def test_reduction_demo_runs_and_reports_execution_boundaries(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()

    output = capsys.readouterr().out
    assert "Executions: 0" in output
    assert "Non-null values by visible column:" in output
    assert "sum(skipna=False): nan" in output
    assert "sum(min_count=4): nan" in output
    assert "Final execution count: 10" in output
    assert "one-query execution boundary" in output
