from __future__ import annotations

import pandas as pd

from scripts.research.run_fast120_fx_currency_strength_scout import Management
from scripts.research.run_fast126_fx_h1_overextension_scout import (
    Policy,
    Signal,
    fast126_status,
    non_overlapping_signals,
    policy_grid,
    side_for_direction,
)


def _signal(entry: str, side: int = 1) -> Signal:
    timestamp = pd.Timestamp(entry)
    return Signal(
        signal_timestamp=timestamp - pd.Timedelta(minutes=1),
        entry_timestamp=timestamp,
        continuation_side=side,
        move_pips=10.0 * side,
        atr_pips=3.0,
        overextension_atr=3.3,
    )


def test_policy_grid_is_deterministic_and_contains_both_directions() -> None:
    policies = policy_grid()
    directions = {policy.direction for policy in policies}

    assert len(policies) == 32
    assert directions == {"continuation", "reversion"}
    assert policies[0].identifier.startswith("h1_continuation_nonoverlap")


def test_non_overlapping_signals_blocks_entries_until_hold_window_ends() -> None:
    signals = (
        _signal("2026-01-01T00:00:00Z"),
        _signal("2026-01-01T02:00:00Z"),
        _signal("2026-01-01T05:00:00Z"),
    )

    selected = non_overlapping_signals(signals, hold_minutes=240)

    assert [signal.entry_timestamp.isoformat() for signal in selected] == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T05:00:00+00:00",
    ]


def test_side_for_direction_follows_or_fades_overextension() -> None:
    signal = _signal("2026-01-01T00:00:00Z", side=1)

    assert side_for_direction(signal, "continuation") == 1
    assert side_for_direction(signal, "reversion") == -1


def test_fast126_status_never_approves_strategy() -> None:
    status, reasons = fast126_status(passed_count=1, finalists_count=1)

    assert status == "FAST126_FX_H1_OVEREXTENSION_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION"
    assert "not_approved_strategy" in reasons
    assert "no_order_execution" in reasons

    status, reasons = fast126_status(passed_count=0, finalists_count=0)

    assert status == "FAST126_FX_H1_OVEREXTENSION_FAILED_SELECTION"
    assert "no_selection_candidate_positive_after_costs" in reasons


def test_policy_shape_documents_operable_non_overlap_trade() -> None:
    policy = Policy(
        identifier="x",
        direction="continuation",
        lookback_bars=8,
        threshold_atr=3.0,
        stop_atr_multiple=1.5,
        max_hold_minutes=480,
        management=Management("fixed_1r", target_r=1.0, activation_r=None, trail_r=None),
    )

    assert policy.max_hold_minutes == 480
    assert policy.management.name == "fixed_1r"
