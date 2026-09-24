from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.research.run_fast167_fast151_external_m1_overlay_intake import (
    EXTERNAL_M1_COLUMNS,
    TARGET_PAIRS,
    build_external_template,
    fast167_status,
    normalize_external_m1_csv,
    write_normalized_pair_files,
)


def _external_rows(timestamp: str = "2026-09-24T12:00:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair": list(TARGET_PAIRS),
            "timestamp": [timestamp] * len(TARGET_PAIRS),
            "bid_open": [1.0] * len(TARGET_PAIRS),
            "bid_high": [1.1] * len(TARGET_PAIRS),
            "bid_low": [0.9] * len(TARGET_PAIRS),
            "bid_close": [1.05] * len(TARGET_PAIRS),
            "ask_open": [1.0001] * len(TARGET_PAIRS),
            "ask_high": [1.1001] * len(TARGET_PAIRS),
            "ask_low": [0.9001] * len(TARGET_PAIRS),
            "ask_close": [1.0501] * len(TARGET_PAIRS),
        }
    )


def test_build_external_template_contains_required_columns_and_target_pairs() -> None:
    template = build_external_template()

    assert set(EXTERNAL_M1_COLUMNS).issubset(template.columns)
    assert template["pair"].tolist() == list(TARGET_PAIRS)


def test_normalize_external_m1_csv_sorts_deduplicates_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "external.csv"
    rows = pd.concat([_external_rows("2026-09-24T12:01:00Z"), _external_rows("2026-09-24T12:00:00Z")])
    rows.to_csv(path, index=False)

    normalized = normalize_external_m1_csv(path)

    assert normalized["pair"].tolist() == [pair for pair in sorted(TARGET_PAIRS) for _ in range(2)]
    assert normalized["timestamp"].min().isoformat() == "2026-09-24T12:00:00+00:00"
    assert normalized["timestamp"].max().isoformat() == "2026-09-24T12:01:00+00:00"


def test_normalize_external_m1_csv_rejects_crossed_quotes(tmp_path: Path) -> None:
    path = tmp_path / "crossed.csv"
    rows = _external_rows()
    rows.loc[0, "ask_open"] = 0.99
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="crossed bid/ask"):
        normalize_external_m1_csv(path)


def test_write_normalized_pair_files_reports_post_preregistration_coverage(tmp_path: Path) -> None:
    path = tmp_path / "external.csv"
    _external_rows().to_csv(path, index=False)
    normalized = normalize_external_m1_csv(path)

    summary = write_normalized_pair_files(normalized, tmp_path / "normalized")

    assert summary["all_target_pairs_present"] is True
    assert summary["all_target_pairs_have_post_preregistration_rows"] is True
    assert set(summary["pairs_with_post_preregistration_rows"]) == set(TARGET_PAIRS)


def test_fast167_status_waits_for_external_input() -> None:
    status, reasons = fast167_status(
        external_m1_supplied=False,
        normalized_summary=None,
        readiness_report=None,
        run_readiness=True,
    )

    assert status == "FAST167_EXTERNAL_M1_INPUT_PENDING"
    assert "external_m1_csv_not_supplied" in reasons
    assert "not_approved_strategy" in reasons
