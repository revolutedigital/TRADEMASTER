import pandas as pd

from scripts.research.run_fast73_anti_stop_density_audit import (
    apply_quote_volume_density_gate,
    density_filter_records,
    fast73_status,
    target_alignment_report,
)


def test_density_filter_counts_recent_candidates_not_only_kept() -> None:
    records = [
        {"decision_time_ms": 1_000, "probability": 0.90},
        {"decision_time_ms": 2_000, "probability": 0.91},
        {"decision_time_ms": 3_000, "probability": 0.92},
        {"decision_time_ms": 310_000, "probability": 0.93},
    ]

    kept = density_filter_records(records, lookback_seconds=300, limit=1)

    assert [row["decision_time_ms"] for row in kept] == [1_000, 2_000, 310_000]
    assert [row["recent_filtered_signal_count"] for row in kept] == [0, 1, 0]


def test_apply_quote_volume_density_gate_filters_low_volume_first() -> None:
    frame = pd.DataFrame(
        {
            "decision_time_ms": [1_000, 2_000, 3_000],
            "probability": [0.9, 0.8, 0.7],
            "quote_volume_30s": [200.0, 50.0, 300.0],
        }
    )

    filtered = apply_quote_volume_density_gate(
        frame,
        quote_volume_threshold=100.0,
        lookback_seconds=300,
        limit=0,
    )

    assert filtered["decision_time_ms"].tolist() == [1_000]


def test_target_alignment_counts_hit_but_lost_cases() -> None:
    frame = pd.DataFrame(
        {
            "paid_stress_before_stop": [True, True, False],
            "stress_net_bps": [-1.0, 2.0, -3.0],
            "first_touch": ["BREAKEVEN", "BREAKEVEN", "STOP"],
        }
    )

    report = target_alignment_report(frame)

    assert report["target_hit_stress_loss_count"] == 1
    assert report["target_miss_stress_win_count"] == 0
    assert report["stop_count"] == 1


def test_fast73_status_fails_extension_before_historical_promotion() -> None:
    status, reasons = fast73_status(
        historical_aggregate={"negative_sufficient_blocks": []},
        extension_aggregate={"negative_sufficient_blocks": ["extension"]},
        sep23={"available": False},
    )

    assert status == "FAST73_ANTI_STOP_DENSITY_FAILED_EXTENSION"
    assert "extension_negative_blocks:1" in reasons
