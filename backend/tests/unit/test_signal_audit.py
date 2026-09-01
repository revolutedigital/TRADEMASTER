"""Tests for durable, trader-readable strategy signal evidence."""

import json
from types import SimpleNamespace

from app.api.v1.signals import _parse_evidence
from app.services.trading_engine import _serialize_signal_evidence


def test_signal_evidence_round_trips_through_the_public_schema() -> None:
    snapshot = _serialize_signal_evidence(
        signal_source="strategy_deployment:44",
        active_strategy_id=44,
        signal_threshold=0.3,
        agreement_ratio=1.0,
        votes=[
            {
                "model": "approved_technical_strategy",
                "action": "BUY",
                "score": 0.8,
                "confidence": 1.0,
            }
        ],
        regime_state=SimpleNamespace(
            market="TRENDING",
            volatility="NORMAL",
            confidence=0.9,
            position_size_mult=0.8,
        ),
        price=100.0,
        atr=2.0,
        atr_pct=0.02,
    )

    evidence = _parse_evidence(snapshot)

    assert evidence is not None
    assert evidence.strategy_deployment_id == 44
    assert evidence.votes[0].model == "approved_technical_strategy"
    assert evidence.votes[0].action == "BUY"
    assert evidence.regime.position_size_multiplier == 0.8
    assert json.loads(snapshot)["price"] == 100.0


def test_signal_evidence_keeps_legacy_or_malformed_rows_readable() -> None:
    assert _parse_evidence(None) is None
    assert _parse_evidence("not-json") is None
    assert _parse_evidence('{"signal_source":"legacy"}') is None
