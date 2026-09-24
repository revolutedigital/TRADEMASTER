from __future__ import annotations

from scripts.research.run_fast175_fast174_regime_gate_audit import (
    fast175_status,
    select_train_preselected_gate,
)


def _gate(
    feature: str,
    *,
    min_train_block_net: float,
    train_net: float,
    validation_passed: bool = False,
    extension_passed: bool = False,
    all_train_blocks: bool = True,
) -> dict:
    return {
        "feature": feature,
        "operator": "gte",
        "quantile": 0.5,
        "train": {
            "passed": True,
            "all_train_blocks_represented": all_train_blocks,
            "min_train_block_net_bps": min_train_block_net,
            "net_total_bps": train_net,
            "trade_count": 12,
        },
        "validation": {"passed": validation_passed},
        "extension": {"passed": extension_passed},
        "strict_passed": validation_passed and extension_passed,
    }


def test_select_train_preselected_gate_uses_train_only_stability_key() -> None:
    unstable_high_net = _gate("high_net", min_train_block_net=-10.0, train_net=1000.0)
    stable_lower_net = _gate("stable", min_train_block_net=5.0, train_net=100.0)
    missing_block = _gate("missing", min_train_block_net=50.0, train_net=200.0, all_train_blocks=False)

    selected = select_train_preselected_gate([unstable_high_net, stable_lower_net, missing_block])

    assert selected is stable_lower_net


def test_fast175_status_marks_posthoc_survivors_as_diagnostic_when_preselected_fails() -> None:
    preselected = _gate("stable", min_train_block_net=5.0, train_net=100.0)

    status, reasons = fast175_status(preselected_gate=preselected, strict_survivor_count=3)

    assert status == "FAST175_POSTHOC_REGIME_GATES_FOUND_BUT_PRESELECTED_GATE_FAILED"
    assert "posthoc_strict_survivors:3" in reasons
    assert "posthoc_survivors_are_diagnostic_only" in reasons


def test_fast175_status_promotes_only_preselected_strict_gate_to_preregistration() -> None:
    preselected = _gate(
        "stable",
        min_train_block_net=5.0,
        train_net=100.0,
        validation_passed=True,
        extension_passed=True,
    )

    status, reasons = fast175_status(preselected_gate=preselected, strict_survivor_count=1)

    assert status == "FAST175_PRESELECTED_REGIME_GATE_REQUIRES_PREREGISTRATION"
    assert "preselected_gate_passed_validation_and_extension" in reasons
