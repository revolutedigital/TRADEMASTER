"""Prospective fixed-gate shadow window orchestration stays bounded."""

from __future__ import annotations

from scripts.research.run_book_fixed_gate_prospective_window import (
    derive_fixed_gate_shadow_window,
)
from scripts.research.run_book_wal_scout import StreamSummary


def test_fixed_gate_window_leaves_history_and_settlement_horizon() -> None:
    streams = {
        "trade": _summary("trade", first_ms=1_000, last_ms=301_000),
        "depth": _summary("depth", first_ms=1_100, last_ms=301_500),
        "spot_trade": _summary("spot_trade", first_ms=1_200, last_ms=300_000),
    }

    window = derive_fixed_gate_shadow_window(
        streams=streams,
        policy=_policy(),
        decision_stride_seconds=5,
        settlement_tail_guard_seconds=2.0,
    )

    assert window.first_decision_time == "1970-01-01T00:00:35Z"
    assert window.last_decision_time == "1970-01-01T00:02:55Z"
    assert window.end.isoformat() == "1970-01-01T00:03:00+00:00"
    assert window.decision_count == 29


def test_fixed_gate_window_requires_spot_stream() -> None:
    streams = {
        "trade": _summary("trade", first_ms=1_000, last_ms=301_000),
        "depth": _summary("depth", first_ms=1_100, last_ms=301_500),
        "spot_trade": _summary("spot_trade", first_ms=None, last_ms=None, row_count=0),
    }

    try:
        derive_fixed_gate_shadow_window(
            streams=streams,
            policy=_policy(),
            decision_stride_seconds=5,
        )
    except ValueError as error:
        assert "missing required streams: spot_trade" in str(error)
    else:
        raise AssertionError("expected missing spot stream to fail closed")


def _policy() -> dict[str, object]:
    return {
        "feature_windows_seconds": (1, 5, 30),
        "horizon_seconds": 120,
    }


def _summary(
    name: str,
    *,
    first_ms: int | None,
    last_ms: int | None,
    row_count: int = 1,
) -> StreamSummary:
    return StreamSummary(
        name=name,
        present=row_count > 0,
        row_count=row_count,
        first_event_time_ms=first_ms,
        last_event_time_ms=last_ms,
        first_event_time=None,
        last_event_time=None,
    )
