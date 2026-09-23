import math

from scripts.research.run_fast80_rare_vol_flow_observation import (
    FLOW_IMBALANCE_5S_THRESHOLD,
    MODEL_SHA256,
    REALIZED_VOL_30S_THRESHOLD,
)
from scripts.research.run_fast84_top_p_silence_diagnostic import (
    diagnostic_summary,
    enrich_diagnostic_row,
    fast84_status,
    nearest_rank_quantile,
    probability_distribution,
)


def test_probability_distribution_uses_nearest_rank_quantiles() -> None:
    distribution = probability_distribution([0.1, 0.4, 0.2, 0.3])

    assert distribution["count"] == 4
    assert distribution["p50"] == 0.2
    assert distribution["p95"] == 0.4
    assert distribution["max"] == 0.4


def test_nearest_rank_quantile_rejects_invalid_quantile() -> None:
    try:
        nearest_rank_quantile([0.1], 0.0)
    except ValueError as error:
        assert "quantile" in str(error)
    else:
        raise AssertionError("expected invalid quantile to fail")


def test_diagnostic_summary_marks_silent_window_far_from_threshold() -> None:
    rows = [
        base_row(probability=0.41, side="BUY"),
        base_row(probability=0.39, side="SELL"),
    ]

    summary = diagnostic_summary(rows, threshold=0.90, near_threshold_margins=(0.05,))
    status, reasons = fast84_status(summary)

    assert status == "FAST84_TOP_P_SILENT_BELOW_THRESHOLD"
    assert summary["would_enter_count"] == 0
    assert math.isclose(summary["distance_from_threshold_at_max_probability"], 0.49)
    assert "threshold_minus_max_probability:0.49000000" in reasons


def test_diagnostic_summary_marks_near_threshold_without_approval() -> None:
    rows = [
        base_row(probability=0.86, side="BUY"),
        base_row(probability=0.83, side="SELL"),
    ]

    summary = diagnostic_summary(rows, threshold=0.90, near_threshold_margins=(0.05,))
    status, reasons = fast84_status(summary)

    assert status == "FAST84_TOP_P_NEAR_THRESHOLD_RESEARCH_ONLY"
    assert summary["would_enter_count"] == 0
    assert summary["near_threshold_counts"]["within_0.05"] == 1
    assert "threshold_minus_max_probability:0.04000000" in reasons


def test_enrich_diagnostic_row_counts_fast80_rule_matches() -> None:
    row = enrich_diagnostic_row(
        base_row(
            probability=0.91,
            side="BUY",
            realized_vol_30s_bps=REALIZED_VOL_30S_THRESHOLD + 1.0,
            flow_imbalance_5s=FLOW_IMBALANCE_5S_THRESHOLD - 0.01,
            would_enter=True,
        )
    )

    summary = diagnostic_summary([row], threshold=0.90, near_threshold_margins=(0.05,))

    assert row["fast80_required_features_present"] is True
    assert row["fast80_rule_match_diagnostic"] is True
    assert summary["fast80_rule_diagnostic"]["rule_match_rows"] == 1
    assert summary["fast80_rule_diagnostic"]["rule_match_would_enter_rows"] == 1


def base_row(
    *,
    probability: float,
    side: str,
    realized_vol_30s_bps: float = 10.0,
    flow_imbalance_5s: float = 0.0,
    would_enter: bool = False,
) -> dict[str, object]:
    return {
        "side": side,
        "probability": probability,
        "threshold": 0.90,
        "would_enter": would_enter,
        "model_sha256": MODEL_SHA256,
        "realized_vol_30s_bps": realized_vol_30s_bps,
        "flow_imbalance_5s": flow_imbalance_5s,
    }
