"""Tamper-evident checks for the research-only shadow signal ledger."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_experiment import ResearchExperimentEvent, ResearchShadowSignal
from app.repositories.research_experiment_repo import (
    research_experiment_repository,
    verify_research_event_chain,
)
from app.services.research.shadow_recorder import (
    shadow_outcome_event_payload,
    shadow_signal_event_payload,
)


@dataclass(frozen=True)
class ResearchShadowLedgerStatus:
    """Verification summary for shadow rows versus hash-chained events."""

    signal_count: int
    outcome_count: int
    signal_event_count: int
    outcome_event_count: int
    verified: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "signal_count": self.signal_count,
            "outcome_count": self.outcome_count,
            "signal_event_count": self.signal_event_count,
            "outcome_event_count": self.outcome_event_count,
            "verified": self.verified,
            "reasons": list(self.reasons),
        }


async def research_shadow_ledger_status(
    db: AsyncSession,
    experiment_id: str,
) -> ResearchShadowLedgerStatus:
    """Verify the official shadow signal/outcome rows against event-chain payloads."""
    signals = await research_experiment_repository.list_shadow_signals(db, experiment_id)
    events = await research_experiment_repository.list_events(db, experiment_id)
    return verify_research_shadow_ledger(signals, events)


def verify_research_shadow_ledger(
    signals: list[ResearchShadowSignal],
    events: list[ResearchExperimentEvent],
) -> ResearchShadowLedgerStatus:
    """Return fail-closed shadow-ledger status without trusting mutable rows alone."""
    reasons: list[str] = []
    event_chain_status = verify_research_event_chain(events)
    if not event_chain_status.verified:
        reasons.append("research_event_chain_unverified")

    signal_events = _events_by_signal_id(events, "SHADOW_SIGNAL_RECORDED", reasons)
    outcome_events = _events_by_signal_id(events, "SHADOW_OUTCOME_RECORDED", reasons)
    seen_signal_ids = {int(signal.id) for signal in signals if signal.id is not None}

    outcome_count = 0
    for signal in signals:
        signal_id = int(signal.id)
        _verify_one_event(
            signal_events,
            signal_id,
            expected=shadow_signal_event_payload(signal),
            missing_reason=f"shadow_signal_{signal_id}_event_missing",
            duplicate_reason=f"shadow_signal_{signal_id}_event_duplicate",
            mismatch_reason=f"shadow_signal_{signal_id}_event_mismatch",
            reasons=reasons,
        )

        outcome_payload = _outcome_payload(signal, reasons)
        if outcome_payload is None:
            if outcome_events.get(signal_id):
                reasons.append(f"shadow_signal_{signal_id}_unexpected_outcome_event")
            continue

        outcome_count += 1
        _verify_one_event(
            outcome_events,
            signal_id,
            expected=shadow_outcome_event_payload(signal, outcome_payload),
            missing_reason=f"shadow_signal_{signal_id}_outcome_event_missing",
            duplicate_reason=f"shadow_signal_{signal_id}_outcome_event_duplicate",
            mismatch_reason=f"shadow_signal_{signal_id}_outcome_event_mismatch",
            reasons=reasons,
        )

    for signal_id in sorted(set(signal_events) - seen_signal_ids):
        reasons.append(f"shadow_signal_{signal_id}_event_without_row")
    for signal_id in sorted(set(outcome_events) - seen_signal_ids):
        reasons.append(f"shadow_signal_{signal_id}_outcome_event_without_row")

    return ResearchShadowLedgerStatus(
        signal_count=len(signals),
        outcome_count=outcome_count,
        signal_event_count=sum(len(rows) for rows in signal_events.values()),
        outcome_event_count=sum(len(rows) for rows in outcome_events.values()),
        verified=not reasons,
        reasons=tuple(reasons),
    )


def _events_by_signal_id(
    events: list[ResearchExperimentEvent],
    kind: str,
    reasons: list[str],
) -> dict[int, list[dict[str, Any]]]:
    by_signal_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.kind != kind:
            continue
        try:
            payload = json.loads(event.payload_json)
        except json.JSONDecodeError:
            reasons.append(f"{kind.lower()}_{event.id}_payload_invalid_json")
            continue
        if not isinstance(payload, dict):
            reasons.append(f"{kind.lower()}_{event.id}_payload_not_object")
            continue
        signal_id = payload.get("signal_id")
        if type(signal_id) is not int or signal_id <= 0:
            reasons.append(f"{kind.lower()}_{event.id}_signal_id_invalid")
            continue
        by_signal_id[signal_id].append(payload)
    return by_signal_id


def _verify_one_event(
    events_by_signal_id: dict[int, list[dict[str, Any]]],
    signal_id: int,
    *,
    expected: dict[str, object],
    missing_reason: str,
    duplicate_reason: str,
    mismatch_reason: str,
    reasons: list[str],
) -> None:
    payloads = events_by_signal_id.get(signal_id, [])
    if not payloads:
        reasons.append(missing_reason)
        return
    if len(payloads) > 1:
        reasons.append(duplicate_reason)
        return
    if payloads[0] != expected:
        reasons.append(mismatch_reason)


def _outcome_payload(
    signal: ResearchShadowSignal,
    reasons: list[str],
) -> dict[str, object] | None:
    if signal.outcome_json is None:
        return None
    try:
        payload = json.loads(signal.outcome_json)
    except json.JSONDecodeError:
        reasons.append(f"shadow_signal_{signal.id}_outcome_json_invalid")
        return None
    if not isinstance(payload, dict):
        reasons.append(f"shadow_signal_{signal.id}_outcome_json_not_object")
        return None
    return payload
