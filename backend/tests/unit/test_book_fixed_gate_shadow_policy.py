"""Fixed book-gate shadow policy scoring stays causal and research-only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.schemas.microstructure import MarketEventType, MicrostructureEvent
from scripts.research.run_book_fixed_gate_shadow_policy import (
    _policy_from_report,
    score_fixed_gate_shadow_events,
)


BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def test_fixed_gate_shadow_policy_selects_sell_when_basis_and_gate_pass() -> None:
    decision_time_ms = int((BASE_TIME + timedelta(milliseconds=100)).timestamp() * 1000)
    policy = _policy()

    selection = score_fixed_gate_shadow_events(
        (
            _book(0),
            _trade(0, 99.0, product="usdm_perpetual"),
            _trade(0, 100.0, product="spot"),
        ),
        policy=policy,
        decision_times_ms=(decision_time_ms,),
        side="SELL",
    )

    assert selection["scored_rows"] == 1
    assert selection["selected_count"] == 1
    decision = selection["decisions"][0]
    assert decision["would_enter"] is True
    assert decision["probability"] == 1.0
    assert decision["threshold"] == 1.0
    assert len(decision["policy_sha256"]) == 64
    assert decision["feature_vector"]["directed_spot_perp_basis_bps"] > 0


def test_fixed_gate_shadow_policy_fails_closed_without_spot() -> None:
    decision_time_ms = int((BASE_TIME + timedelta(milliseconds=100)).timestamp() * 1000)

    selection = score_fixed_gate_shadow_events(
        (
            _book(0),
            _trade(0, 99.0, product="usdm_perpetual"),
        ),
        policy=_policy(),
        decision_times_ms=(decision_time_ms,),
        side="SELL",
        include_non_entries=True,
    )

    assert selection["selected_count"] == 0
    assert selection["decisions"][0]["would_enter"] is False
    assert "spot_unavailable" in selection["decisions"][0]["decision_reasons"]


def test_policy_from_report_requires_research_only_boundary() -> None:
    report = _policy_report()
    report["execution_authorization"] = "testnet"

    try:
        _policy_from_report(report)
    except ValueError as error:
        assert "execution_authorization=none" in str(error)
    else:
        raise AssertionError("expected policy boundary validation to fail")


def _policy() -> dict[str, object]:
    return _policy_from_report(_policy_report())


def _policy_report() -> dict[str, object]:
    return {
        "research_only": True,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "approved_strategy": False,
        "experiment_stage": "book_regime_gate_confirmation",
        "config": {
            "horizons_seconds": (120,),
            "expected_cost_bps": 5.0,
            "initial_stop_bps": 20.0,
            "feature_windows_seconds": (1, 5, 30),
            "max_book_staleness_ms": 1_000,
        },
        "base_thresholds": [
            {
                "feature": "directed_spot_perp_basis_bps",
                "direction": "high",
                "threshold": 5.0,
            }
        ],
        "frozen_gate": {
            "feature": "spot_update_age_ms",
            "direction": "low",
            "threshold": 500.0,
        },
    }


def _trade(offset_ms: int, price: float, *, product: str) -> MicrostructureEvent:
    return MicrostructureEvent(
        product=product,
        symbol="BTCUSDT",
        event_type=MarketEventType.TRADE,
        event_time=BASE_TIME + timedelta(milliseconds=offset_ms),
        price=price,
        quantity=1.0,
        quote_quantity=price,
        is_buyer_maker=True,
    )


def _book(offset_ms: int) -> MicrostructureEvent:
    return MicrostructureEvent(
        product="usdm_perpetual",
        symbol="BTCUSDT",
        event_type=MarketEventType.DEPTH,
        event_time=BASE_TIME + timedelta(milliseconds=offset_ms),
        bid_price=98.99,
        bid_quantity=1.0,
        ask_price=99.01,
        ask_quantity=1.0,
    )
