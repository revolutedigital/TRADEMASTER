"""Contract checks for the research-only microstructure control plane."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = PROJECT_ROOT / "docs" / "openapi" / "microstructure-research.yaml"
PROTOCOL_PATH = (
    PROJECT_ROOT / "docs" / "research" / "microstructure-v1-preregistration.md"
)
PROTOCOL_HASH_PATH = PROTOCOL_PATH.with_suffix(".sha256")


def _load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def test_microstructure_spec_is_openapi_31_and_research_only() -> None:
    spec = _load_spec()

    assert spec["openapi"] == "3.1.0"
    assert spec["servers"] == [{"url": "/api/v1"}]

    paths = set(spec["paths"])
    assert paths == {
        "/research/microstructure/evidence-gate",
        "/research/microstructure/experiments",
        "/research/microstructure/experiments/{experiment_id}",
        "/research/microstructure/experiments/{experiment_id}/testnet-eligibility",
        "/research/microstructure/experiments/{experiment_id}/shadow-signals",
        "/research/microstructure/shadow-signals/{signal_id}/outcome",
        "/research/microstructure/experiments/{experiment_id}/freeze",
        "/research/microstructure/experiments/{experiment_id}/report",
    }
    forbidden_route_terms = {"order", "deploy", "activate", "engine", "arm", "execution"}
    assert not any(term in path.lower() for path in paths for term in forbidden_route_terms)

    safety = spec["components"]["schemas"]["SafetyBoundary"]["properties"]
    assert safety["research_only"]["const"] is True
    assert safety["order_submission_allowed"]["const"] is False
    assert safety["execution_authorization"]["const"] == "none"


def test_microstructure_spec_freezes_the_approved_v1_product() -> None:
    product = _load_spec()["components"]["schemas"]["ProductContract"]["properties"]

    assert product["execution_venue"]["const"] == "binance_usdm_futures"
    assert product["execution_product"]["const"] == "perpetual"
    assert product["symbol"]["const"] == "BTCUSDT"
    assert product["directions"]["const"] == ["LONG", "SHORT"]
    assert product["auxiliary_signal_venue"]["const"] == "binance_spot"
    assert product["horizons_seconds"]["const"] == [5, 15, 30, 120, 300]
    assert product["position_model"]["const"] == "one_net_position"


def test_microstructure_spec_publishes_the_approval_gate() -> None:
    gate = _load_spec()["components"]["schemas"]["ApprovalGate"]["properties"]

    assert gate["min_oos_folds"]["const"] == 3
    assert gate["min_oos_portfolio_trades"]["const"] == 200
    assert gate["min_oos_utc_days"]["const"] == 20
    assert gate["adjusted_one_sided_confidence"]["const"] == 0.95
    assert gate["max_probability_backtest_overfitting"]["const"] == 0.2
    assert gate["book_evidence_min_complete_days"]["const"] == 60
    assert gate["prospective_shadow_min_days"]["const"] == 20
    assert gate["prospective_shadow_max_days"]["const"] == 30
    assert gate["top_p_tails_pct"]["const"] == [1, 2, 5, 10]

    cost = _load_spec()["components"]["schemas"]["CostProfile"]["properties"]
    assert cost["stress_roundtrip_bps"]["minimum"] == 20
    assert cost["default_order_style"]["const"] == "marketable_taker"


def test_microstructure_spec_publishes_the_book_evidence_artifact_contract() -> None:
    schemas = _load_spec()["components"]["schemas"]
    status = schemas["EvidenceGateStatus"]
    gate = schemas["BookEvidenceGate"]["properties"]

    assert status["additionalProperties"] is False
    assert "book_evidence_gate" in status["required"]
    assert "safety" in status["required"]
    assert gate["required_complete_days"]["const"] == 60
    assert gate["manifest_sha256"]["pattern"] == "^[a-f0-9]{64}$"


def test_microstructure_spec_publishes_metadata_only_testnet_eligibility() -> None:
    eligibility = _load_spec()["components"]["schemas"]["TestnetEligibility"]
    properties = eligibility["properties"]

    assert eligibility["additionalProperties"] is False
    assert properties["explicit_testnet_release"]["const"] is False
    assert properties["release_request_required"]["const"] is True
    assert "prospective_shadow_outcome_days" in eligibility["required"]
    assert "prospective_shadow_outcome_signal_count" in eligibility["required"]
    assert "prospective_shadow_positive" in eligibility["required"]
    assert properties["order_submission_allowed"]["const"] is False
    assert properties["execution_authorization"]["const"] == "none"


def test_microstructure_spec_publishes_research_only_shadow_signal_contract() -> None:
    schemas = _load_spec()["components"]["schemas"]
    shadow_signal = schemas["ShadowSignal"]

    assert shadow_signal["additionalProperties"] is False
    assert "feature_vector_sha256" in shadow_signal["required"]
    assert "outcome_recorded" in shadow_signal["required"]
    assert "safety" in shadow_signal["required"]
    assert "order_id" not in shadow_signal["properties"]
    assert "execution_authorization" not in shadow_signal["properties"]


def test_preregistration_hash_matches_the_frozen_document() -> None:
    expected_hash = PROTOCOL_HASH_PATH.read_text(encoding="utf-8").strip()
    actual_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()

    assert len(expected_hash) == 64
    assert expected_hash == actual_hash
