from __future__ import annotations

import pandas as pd

from scripts.research.run_fast146_usd_quoted_major_basket_audit import (
    ENTRY_POLICY_ID,
    PRE_SAMPLE_LABEL,
    RISK_POLICY_ID,
    evaluate_groups,
    fast146_status,
    filtered_sample,
    run,
    summarize_groups,
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
        "candidate_id": f"{pair}:{ENTRY_POLICY_ID}:{RISK_POLICY_ID}",
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
        for day in range(1, 9)
    ]


def test_usd_quoted_major_group_is_selected_on_pre_and_audited_current() -> None:
    pre_rows = []
    current_rows = []
    for pair in ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=40.0))
        current_rows.extend(_rows(sample=None, pair=pair, start_year=2019, stress=30.0))
    for pair in ("EURJPY", "GBPJPY", "USDJPY"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=-20.0))
        current_rows.extend(_rows(sample=None, pair=pair, start_year=2019, stress=-20.0))

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

    summary = summarize_groups(evaluate_groups(pre_sample, current_sample))
    status, reasons = fast146_status(summary)

    assert summary["selected_by_pre_group_id"] == "usd_quoted_majors"
    assert summary["target_group"]["both_passed"] is True
    assert status == "FAST146_USD_QUOTED_MAJOR_BASKET_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons


def test_current_only_group_does_not_become_selected_candidate() -> None:
    pre_rows = []
    current_rows = []
    for pair in ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=-10.0))
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

    summary = summarize_groups(evaluate_groups(pre_sample, current_sample))
    status, _reasons = fast146_status(summary)

    assert summary["selected_by_pre_group_id"] is None
    assert status == "FAST146_GROUP_BASKET_FAILED_PRE_SELECTION"


def test_run_writes_research_only_report(tmp_path) -> None:
    pre_rows = []
    current_rows = []
    for pair in ("AUDUSD", "EURUSD", "GBPUSD", "NZDUSD"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=40.0))
        current_rows.extend(_rows(sample=None, pair=pair, start_year=2019, stress=30.0))
    for pair in ("EURJPY", "GBPJPY", "USDJPY"):
        pre_rows.extend(_rows(sample=PRE_SAMPLE_LABEL, pair=pair, start_year=2012, stress=-20.0))
        current_rows.extend(_rows(sample=None, pair=pair, start_year=2019, stress=-20.0))
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
    assert report["summary"]["selected_by_pre_group_id"] == "usd_quoted_majors"
    assert (tmp_path / "out" / "group-basket-metrics.csv").exists()
