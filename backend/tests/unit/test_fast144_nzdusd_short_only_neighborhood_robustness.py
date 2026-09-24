from __future__ import annotations

import pandas as pd

from scripts.research.run_fast144_nzdusd_short_only_neighborhood_robustness import (
    FROZEN_CANDIDATE_ID,
    PRE_SAMPLE_LABEL,
    candidate_grid,
    candidate_id,
    evaluate_candidates,
    fast144_status,
    filtered_sample,
    run,
    summarize_neighborhood,
)


def _trade(
    *,
    sample: str | None,
    channel_days: int,
    hold_days: int,
    stop_atr_multiple: float,
    year: int,
    day: int,
    stress: float,
    base: float | None = None,
) -> dict:
    timestamp = pd.Timestamp(f"{year}-01-{day:02d}T00:00:00Z")
    row = {
        "candidate_id": candidate_id(channel_days, hold_days, stop_atr_multiple),
        "pair": "NZDUSD",
        "entry_policy_id": f"d1_donchian_fade_ch{channel_days}_hold{hold_days}",
        "risk_policy_id": f"stop{stop_atr_multiple:g}_fixed_1r",
        "channel_days": channel_days,
        "hold_days": hold_days,
        "stop_atr_multiple": stop_atr_multiple,
        "management": "fixed_1r",
        "entry_timestamp": timestamp,
        "side": "SHORT",
        "net_base_pips": stress if base is None else base,
        "net_stress_pips": stress,
    }
    if sample is not None:
        row["sample"] = sample
    return row


def _passing_rows(
    *,
    sample: str | None,
    channel_days: int,
    hold_days: int,
    stop_atr_multiple: float,
    start_year: int,
    stress: float,
) -> list[dict]:
    return [
        _trade(
            sample=sample,
            channel_days=channel_days,
            hold_days=hold_days,
            stop_atr_multiple=stop_atr_multiple,
            year=start_year + year_offset,
            day=day,
            stress=stress,
        )
        for year_offset in range(3)
        for day in range(1, 5)
    ]


def test_candidate_grid_marks_exact_and_local_neighborhood() -> None:
    grid = candidate_grid()

    assert len(grid) == 27
    assert grid["is_exact_frozen_candidate"].sum() == 1
    assert grid["is_local_neighborhood"].sum() == 8
    assert grid.loc[grid["candidate_id"].eq(FROZEN_CANDIDATE_ID), "grid_distance_from_frozen"].iloc[0] == 0


def test_neighborhood_status_replicates_but_stays_narrow() -> None:
    pre_rows = []
    current_rows = []
    for stop in (1.5, 2.0):
        pre_rows.extend(
            _passing_rows(
                sample=PRE_SAMPLE_LABEL,
                channel_days=100,
                hold_days=20,
                stop_atr_multiple=stop,
                start_year=2012,
                stress=50.0,
            )
        )
        current_rows.extend(
            _passing_rows(
                sample=None,
                channel_days=100,
                hold_days=20,
                stop_atr_multiple=stop,
                start_year=2019,
                stress=25.0,
            )
        )

    # A nearby channel/hold shape has enough trades but negative expectancy, so
    # the audit must not report broad robustness.
    pre_rows.extend(
        _passing_rows(
            sample=PRE_SAMPLE_LABEL,
            channel_days=55,
            hold_days=20,
            stop_atr_multiple=2.0,
            start_year=2012,
            stress=-10.0,
        )
    )
    current_rows.extend(
        _passing_rows(
            sample=None,
            channel_days=55,
            hold_days=20,
            stop_atr_multiple=2.0,
            start_year=2019,
            stress=-10.0,
        )
    )

    pre_sample = filtered_sample(
        pd.DataFrame(pre_rows),
        sample_column="sample",
        sample_value=PRE_SAMPLE_LABEL,
    )
    current_sample = filtered_sample(
        pd.DataFrame(current_rows),
        sample_column=None,
        sample_value=None,
    )

    summary = summarize_neighborhood(evaluate_candidates(pre_sample, current_sample))
    status, reasons = fast144_status(summary)

    assert summary["exact_candidate"]["both_passed"]
    assert summary["both_pass_count"] == 2
    assert summary["local_both_pass_count"] == 2
    assert status == "FAST144_NZDUSD_SHORT_ONLY_EDGE_REPLICATED_BUT_NARROW_REQUIRES_OBSERVATION"
    assert "not_approved_strategy" in reasons


def test_run_writes_research_only_report(tmp_path) -> None:
    pre_rows = _passing_rows(
        sample=PRE_SAMPLE_LABEL,
        channel_days=100,
        hold_days=20,
        stop_atr_multiple=2.0,
        start_year=2012,
        stress=50.0,
    )
    current_rows = _passing_rows(
        sample=None,
        channel_days=100,
        hold_days=20,
        stop_atr_multiple=2.0,
        start_year=2019,
        stress=25.0,
    )
    pre_path = tmp_path / "pre.parquet"
    current_path = tmp_path / "current.parquet"
    pd.DataFrame(pre_rows).to_parquet(pre_path, index=False)
    pd.DataFrame(current_rows).to_parquet(current_path, index=False)

    report = run(
        pre_input_path=pre_path,
        current_input_path=current_path,
        output_root=tmp_path / "out",
    )

    assert report["research_only"] is True
    assert report["approved_strategy"] is False
    assert report["orders_sent"] is False
    assert report["summary"]["exact_candidate"]["both_passed"] is True
    assert report["summary"]["both_pass_count"] == 1
    assert (tmp_path / "out" / "candidate-neighborhood-metrics.csv").exists()
