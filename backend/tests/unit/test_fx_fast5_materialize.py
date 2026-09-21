"""Round-5 materialization boundaries and alignment helpers."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.research import fx_fast5_materialize as materialize


def test_expected_feature_index_reads_only_requested_year(tmp_path: Path) -> None:
    index = pd.DatetimeIndex(["2020-12-31T23:00:00Z", "2021-01-01T00:00:00Z"])
    pd.DataFrame({"decision_index": [1, 2]}, index=index).to_parquet(
        tmp_path / "EURUSD-features.parquet"
    )
    result = materialize.expected_feature_index(tmp_path, "EURUSD", 2021)
    assert result.tolist() == [index[1]]


def test_materializer_refuses_2022_before_any_file_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2022 and protected years are closed"):
        materialize.materialize(("EURUSD",), (2022,), tmp_path, tmp_path)
