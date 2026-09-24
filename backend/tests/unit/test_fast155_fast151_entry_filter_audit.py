from __future__ import annotations

import pandas as pd

from scripts.research.run_fast155_fast151_entry_filter_audit import (
    EntryFilter,
    add_entry_features,
    fast155_status,
    filter_mask,
    select_filter,
)


def test_add_entry_features_computes_breakout_r() -> None:
    trades = pd.DataFrame({"breakout_distance_pips": [10.0], "risk_pips": [50.0], "daily_atr_pips": [25.0]})

    enriched = add_entry_features(trades)

    assert enriched.iloc[0]["breakout_r"] == 0.2
    assert enriched.iloc[0]["atr_pips"] == 25.0


def test_filter_mask_supports_between_thresholds() -> None:
    trades = pd.DataFrame({"atr_pips": [10.0, 20.0, 30.0]})

    mask = filter_mask(
        trades,
        EntryFilter(
            "atr_between",
            feature="atr_pips",
            operator="between",
            lower_threshold=15.0,
            upper_threshold=25.0,
        ),
    )

    assert mask.tolist() == [False, True, False]


def test_select_filter_keeps_baseline_when_no_candidate_improves_pre_lcb() -> None:
    selected = select_filter(
        [
            {
                "filter_id": "no_extra_filter",
                "pre_passed_gate": True,
                "pre": {"lower_stress_pips": 10.0, "trade_count": 50},
            },
            {
                "filter_id": "weaker_filter",
                "pre_passed_gate": True,
                "pre": {"lower_stress_pips": 9.0, "trade_count": 60},
            },
        ]
    )

    assert selected is not None
    assert selected["filter_id"] == "no_extra_filter"


def test_fast155_status_rejects_simple_top_p_proxies_when_baseline_selected() -> None:
    status, reasons = fast155_status({"filter_id": "no_extra_filter"})

    assert status == "FAST155_NO_ENTRY_FILTER_IMPROVES_FAST151_PRE_GATE"
    assert "simple_top_p_proxy_filters_rejected" in reasons
    assert "not_approved_strategy" in reasons
