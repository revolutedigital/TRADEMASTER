from __future__ import annotations

import pandas as pd

from scripts.research.run_fast135_nzdusd_donchian_fade_fixed1r_temporal_audit import (
    BAR_COLUMNS,
    fast135_status,
    raw_pair_frame,
)


def _summary(slice_name: str, *, stress_lcb: float | None) -> dict:
    return {
        "slice": slice_name,
        "stress": {
            "lower_confidence_bound_pips": stress_lcb,
        },
    }


def test_fast135_status_reports_mixed_temporal_stability() -> None:
    status, reasons = fast135_status(
        [
            _summary("pre_2011_2018", stress_lcb=-1.0),
            _summary("current_2019_2026", stress_lcb=2.0),
        ]
    )

    assert status == "FAST135_TEMPORAL_AUDIT_MIXED_STABILITY_REQUIRES_PROSPECTIVE_CONFIRMATION"
    assert "not_approved_strategy" in reasons
    assert "pre_2019_lcb_not_positive" in reasons


def test_fast135_status_reports_both_samples_positive_without_approving() -> None:
    status, reasons = fast135_status(
        [
            _summary("pre_2011_2018", stress_lcb=1.0),
            _summary("current_2019_2026", stress_lcb=2.0),
        ]
    )

    assert status == "FAST135_TEMPORAL_AUDIT_BOTH_SAMPLES_POSITIVE_REQUIRES_PROSPECTIVE_CONFIRMATION"
    assert "not_approved_strategy" in reasons
    assert "prospective_confirmation_still_required" in reasons


def test_raw_pair_frame_drops_crossed_quotes(tmp_path) -> None:
    index = pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"], name="timestamp")
    frame = pd.DataFrame(
        [
            [1.0, 1.1, 0.9, 1.0, 1.2, 1.3, 1.1, 1.2],
            [1.0, 1.1, 0.9, 1.0, 0.8, 0.9, 0.7, 0.8],
        ],
        columns=BAR_COLUMNS,
        index=index,
    )
    frame.to_parquet(tmp_path / "NZDUSD_M1_202601_202601.parquet")

    loaded = raw_pair_frame(tmp_path, "NZDUSD")

    assert len(loaded) == 1
    assert loaded.attrs["raw_bars"] == 2
    assert loaded.attrs["dropped_crossed_quote_bars"] == 1
