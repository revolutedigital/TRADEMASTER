from __future__ import annotations

import pandas as pd

from scripts.research.run_fast145_nzdusd_short_only_cross_pair_replication import (
    ENTRY_POLICY_ID,
    PRE_SAMPLE_LABEL,
    RISK_POLICY_ID,
    evaluate_pairs,
    fast145_status,
    filtered_sample,
    pair_candidate_id,
    run,
    summarize_replication,
)


def _trade(
    *,
    sample: str | None,
    pair: str,
    year: int,
    day: int,
    stress: float,
    base: float | None = None,
) -> dict:
    timestamp = pd.Timestamp(f"{year}-01-{day:02d}T00:00:00Z")
    row = {
        "candidate_id": pair_candidate_id(pair),
        "pair": pair,
        "entry_policy_id": ENTRY_POLICY_ID,
        "risk_policy_id": RISK_POLICY_ID,
        "management": "fixed_1r",
        "entry_timestamp": timestamp,
        "side": "SHORT",
        "net_base_pips": stress if base is None else base,
        "net_stress_pips": stress,
    }
    if sample is not None:
        row["sample"] = sample
    return row


def _rows(
    *,
    sample: str | None,
    pair: str,
    start_year: int,
    stress: float,
) -> list[dict]:
    return [
        _trade(sample=sample, pair=pair, year=start_year + year_offset, day=day, stress=stress)
        for year_offset in range(3)
        for day in range(1, 5)
    ]


def test_cross_pair_summary_does_not_count_current_only_pair_as_replication() -> None:
    pre_rows = _rows(sample=PRE_SAMPLE_LABEL, pair="NZDUSD", start_year=2012, stress=40.0)
    current_rows = _rows(sample=None, pair="NZDUSD", start_year=2019, stress=30.0)

    # AUDUSD passes current but fails pre, so it is only a lead, not replication.
    pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair="AUDUSD", start_year=2012, stress=-10.0))
    current_rows.extend(_rows(sample=None, pair="AUDUSD", start_year=2019, stress=30.0))

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

    summary = summarize_replication(evaluate_pairs(pre_sample, current_sample))
    status, reasons = fast145_status(summary)

    assert summary["passed_pairs"] == ["NZDUSD"]
    assert summary["current_only_pairs"] == ["AUDUSD"]
    assert summary["non_anchor_pass_count"] == 0
    assert status == "FAST145_NZDUSD_SHORT_ONLY_CROSS_PAIR_NOT_REPLICATED_REQUIRES_OBSERVATION"
    assert "not_approved_strategy" in reasons


def test_cross_pair_status_detects_non_anchor_replication() -> None:
    pre_rows = []
    current_rows = []
    for pair in ("AUDUSD", "NZDUSD"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=40.0))
        current_rows.extend(_rows(sample=None, pair=pair, start_year=2019, stress=30.0))

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

    summary = summarize_replication(evaluate_pairs(pre_sample, current_sample))
    status, _reasons = fast145_status(summary)

    assert summary["non_anchor_pass_count"] == 1
    assert status == "FAST145_NZDUSD_SHORT_ONLY_CROSS_PAIR_REPLICATED_REQUIRES_OBSERVATION"


def test_run_writes_research_only_report(tmp_path) -> None:
    pre_path = tmp_path / "pre.parquet"
    current_path = tmp_path / "current.parquet"
    pd.DataFrame(
        _rows(sample=PRE_SAMPLE_LABEL, pair="NZDUSD", start_year=2012, stress=40.0)
    ).to_parquet(pre_path, index=False)
    pd.DataFrame(_rows(sample=None, pair="NZDUSD", start_year=2019, stress=30.0)).to_parquet(
        current_path,
        index=False,
    )

    report = run(
        pre_input_path=pre_path,
        current_input_path=current_path,
        output_root=tmp_path / "out",
    )

    assert report["research_only"] is True
    assert report["approved_strategy"] is False
    assert report["orders_sent"] is False
    assert report["summary"]["passed_pairs"] == ["NZDUSD"]
    assert (tmp_path / "out" / "cross-pair-metrics.csv").exists()
