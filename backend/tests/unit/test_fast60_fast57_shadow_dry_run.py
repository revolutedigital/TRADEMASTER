from datetime import UTC, datetime, timedelta

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from scripts.research.run_fast60_fast57_shadow_dry_run import (
    align_datetime_to_stride,
    apply_fast57_density_gate,
    book_context_for_decision,
    build_book_context,
    fast60_run_status,
    fast60_status,
)


def test_density_gate_keeps_only_first_signal_inside_lookback() -> None:
    base_ms = 1_800_000_000_000
    decisions = [
        _decision(base_ms, probability=0.91, side="BUY"),
        _decision(base_ms, probability=0.95, side="SELL"),
        _decision(base_ms + 299_000, probability=0.99, side="BUY"),
        _decision(base_ms + 601_000, probability=0.92, side="BUY"),
    ]

    gated = apply_fast57_density_gate(decisions, lookback_seconds=300, limit=0)

    assert [row["side"] for row in gated] == ["SELL", "BUY"]
    assert [row["recent_signal_count"] for row in gated] == [0, 0]


def test_align_datetime_to_stride_ceilings_to_next_utc_boundary() -> None:
    value = datetime(2026, 1, 1, 0, 0, 3, 250_000, tzinfo=UTC)

    assert align_datetime_to_stride(value, stride_seconds=5) == datetime(
        2026,
        1,
        1,
        0,
        0,
        5,
        tzinfo=UTC,
    )


def test_book_context_uses_latest_depth_before_decision() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    context = build_book_context(
        (
            _depth(start, bid=100.0, ask=100.1),
            _depth(start + timedelta(seconds=1), bid=101.0, ask=101.1),
        )
    )
    decision_ms = int((start + timedelta(seconds=1, milliseconds=250)).timestamp() * 1000)

    row = book_context_for_decision(context, decision_ms)

    assert row["book_available"] is True
    assert row["book_age_ms"] == 250
    assert row["spread_bps"] is not None
    assert row["top_bid_notional_usdt"] == 101_000.0


def test_fast60_statuses_are_fail_closed() -> None:
    assert fast60_run_status(raw_selected_count=0, gated_selected_count=0)[0] == (
        "FAST60_SHADOW_DRY_RUN_NO_TOP_P_SIGNALS"
    )
    assert fast60_run_status(raw_selected_count=3, gated_selected_count=0)[0] == (
        "FAST60_SHADOW_DRY_RUN_DENSITY_FILTERED_ALL"
    )
    status, reasons = fast60_status(
        {
            "wal_root_count": 1,
            "run_count": 1,
            "raw_selected_count": 5,
            "gated_selected_count": 2,
            "runs_with_gated_signals": 1,
        }
    )

    assert status == "FAST60_GATED_SHADOW_SIGNALS_RESEARCH_ONLY"
    assert "not_approved_strategy" in reasons


def _decision(decision_time_ms: int, *, probability: float, side: str) -> dict[str, object]:
    return {
        "label": "test",
        "decision_time_ms": decision_time_ms,
        "decision_time": datetime.fromtimestamp(decision_time_ms / 1000, tz=UTC).isoformat(),
        "side": side,
        "probability": probability,
        "threshold": 0.90,
    }


def _depth(event_time: datetime, *, bid: float, ask: float) -> MicrostructureEvent:
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.DEPTH,
        event_time=event_time,
        bid_price=bid,
        bid_quantity=1000.0,
        ask_price=ask,
        ask_quantity=1000.0,
    )
