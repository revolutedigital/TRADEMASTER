from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.run_fast125_usdjpy_compression_topp_model import (
    add_entry_features,
    evaluate_gates,
    fast125_status,
    metric_summary,
    threshold_sample,
    train_candidate_is_eligible,
)


def _row(
    *,
    entry_timestamp: str,
    side: str = "LONG",
    net_base_pips: float = 2.0,
    net_stress_pips: float = 1.5,
) -> dict:
    return {
        "pair": "USDJPY",
        "policy_id": "tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r",
        "entry_timestamp": entry_timestamp,
        "side": side,
        "breakout_distance_pips": 2.0,
        "atr_pips": 10.0,
        "risk_pips": 15.0,
        "compression_value_pips": 4.0,
        "compression_threshold_pips": 8.0,
        "net_base_pips": net_base_pips,
        "net_stress_pips": net_stress_pips,
    }


def test_add_entry_features_uses_only_entry_known_fields() -> None:
    frame = add_entry_features(
        pd.DataFrame(
            [
                _row(entry_timestamp="2026-09-24T04:50:13Z", side="LONG"),
                _row(entry_timestamp="2026-09-25T05:50:13Z", side="SHORT"),
            ]
        )
    )

    assert frame["side_num"].tolist() == [1, 0]
    assert frame["compression_ratio"].tolist() == [0.5, 0.5]
    assert frame["breakout_over_atr"].tolist() == [0.2, 0.2]
    assert {"hour_sin", "weekday_cos", "month_sin"}.issubset(frame.columns)


def test_threshold_sample_applies_train_threshold_to_live_scores() -> None:
    frame = pd.DataFrame([_row(entry_timestamp=f"2027-01-0{index + 1}T00:00:00Z") for index in range(3)])
    scores = np.array([0.1, 0.5, 0.9])

    selected = threshold_sample(frame, scores, threshold=0.5)

    assert len(selected) == 2
    assert selected["model_score"].tolist() == [0.5, 0.9]


def test_metric_summary_and_train_eligibility() -> None:
    rows = []
    for index in range(80):
        rows.append(
            _row(
                entry_timestamp=f"{2027 + index // 40}-01-{(index % 28) + 1:02d}T00:00:00Z",
                net_base_pips=2.0,
                net_stress_pips=1.5,
            )
        )
    summary = metric_summary(pd.DataFrame(rows), alpha=0.05)

    assert summary["trade_count"] == 80
    assert summary["positive_year_share"] == 1.0
    assert train_candidate_is_eligible(summary) is True


def test_fast125_status_requires_lift_and_positive_lower_bound() -> None:
    selected = {
        "trade_count": 333,
        "mean_stress_pips": 2.1,
        "lower_confidence_bound_pips": -0.1,
        "positive_year_share": 0.8,
        "minimum_yearly_trades": 10,
        "worst_year_mean_stress_pips": -1.0,
    }
    baseline = {"mean_stress_pips": 1.9}

    gates = evaluate_gates(selected_summary=selected, baseline_summary=baseline)
    status, reasons = fast125_status(gates)

    assert gates["shadow_candidate"] is False
    assert status == "FAST125_USDJPY_COMPRESSION_TOPP_MODEL_FAILED_OOS"
    assert "selected_stress_lower_confidence_bound_not_positive" in reasons
    assert "stress_lift_vs_unfiltered_baseline_too_small" in reasons
