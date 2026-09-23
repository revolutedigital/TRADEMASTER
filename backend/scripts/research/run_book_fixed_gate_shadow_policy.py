"""Score the frozen spot-age book gate as research-only shadow decisions.

This is the online-style counterpart to the executable replay. It consumes
chronological microstructure events, derives causal features at each decision
time, applies the frozen rule, and writes a dry-run report. It never submits
orders and does not write to the research ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.schemas.microstructure import MicrostructureEvent  # noqa: E402
from app.services.research.microstructure_features import (  # noqa: E402
    CausalMicrostructureFeatureEngine,
    feature_vector_for_side,
)
from scripts.research.run_online_shadow_policy import (  # noqa: E402
    EventRoots,
    decision_times_between,
    load_online_microstructure_events,
)


POLICY_FEATURE_COLUMNS = (
    "book_available",
    "book_update_age_ms",
    "spread_bps",
    "spot_available",
    "spot_update_age_ms",
    "spot_perp_basis_bps",
    "directed_spot_perp_basis_bps",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--wal-root", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", type=_parse_datetime, required=True)
    parser.add_argument("--end", type=_parse_datetime, required=True)
    parser.add_argument("--decision-stride-seconds", type=int, default=5)
    parser.add_argument("--side", choices=("BUY", "SELL"), default="SELL")
    parser.add_argument("--include-non-entries", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "backend"
        / "data"
        / "microstructure_v1"
        / "reports"
        / "book-fixed-gate-shadow-policy",
    )
    arguments = parser.parse_args()
    if arguments.end <= arguments.start:
        parser.error("--end must be after --start")
    if arguments.decision_stride_seconds <= 0:
        parser.error("--decision-stride-seconds must be positive")

    report = run_book_fixed_gate_shadow_policy(
        policy_report_path=arguments.policy_report,
        wal_root=arguments.wal_root,
        symbol=arguments.symbol.upper(),
        start=arguments.start,
        end=arguments.end,
        decision_stride_seconds=arguments.decision_stride_seconds,
        side=arguments.side,
        include_non_entries=arguments.include_non_entries,
        output_root=arguments.output_root,
    )
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))  # noqa: T201
    return 0


def run_book_fixed_gate_shadow_policy(
    *,
    policy_report_path: Path,
    wal_root: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    decision_stride_seconds: int = 5,
    side: str = "SELL",
    include_non_entries: bool = False,
    output_root: Path,
) -> dict[str, object]:
    policy_report = json.loads(policy_report_path.read_text(encoding="utf-8"))
    policy = _policy_from_report(policy_report)
    decision_times_ms = decision_times_between(
        start,
        end,
        stride_seconds=decision_stride_seconds,
    )
    loaded = load_online_microstructure_events(
        artifact={"feature_columns": list(POLICY_FEATURE_COLUMNS)},
        symbol=symbol,
        start=start,
        end=end,
        history_seconds=max(policy["feature_windows_seconds"]),
        roots=EventRoots(
            trade=wal_root / "trade",
            book=wal_root / "depth",
            mark=None,
            liquidation=None,
            spot=wal_root / "spot_trade",
        ),
    )
    selection = score_fixed_gate_shadow_events(
        loaded.events,
        policy=policy,
        decision_times_ms=decision_times_ms,
        side=side,
        include_non_entries=include_non_entries,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    decisions_path = output_root / "shadow_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for decision in selection["decisions"]:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")
    return _write_report(
        output_root,
        {
            "research_only": True,
            "order_submission_allowed": False,
            "execution_authorization": "none",
            "committed": False,
            "dry_run": True,
            "experiment_stage": "book_fixed_gate_shadow_policy",
            "policy_report_path": str(policy_report_path),
            "policy_sha256": _sha256(policy),
            "symbol": symbol,
            "decision_interval": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "decision_stride_seconds": decision_stride_seconds,
            "decision_count": len(decision_times_ms),
            "loaded_interval": {
                "start": loaded.load_start.isoformat(),
                "end": loaded.load_end.isoformat(),
            },
            "event_counts": loaded.counts,
            "event_roots": loaded.roots.as_report(),
            "policy": policy,
            "scored_rows": selection["scored_rows"],
            "selected_count": selection["selected_count"],
            "decisions_path": str(decisions_path),
            "decisions": selection["decisions"],
        },
    )


def score_fixed_gate_shadow_events(
    events: tuple[MicrostructureEvent, ...],
    *,
    policy: dict[str, Any],
    decision_times_ms: tuple[int, ...],
    side: str = "SELL",
    include_non_entries: bool = False,
) -> dict[str, object]:
    normalized_side = side.upper()
    if normalized_side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")
    if any(later < earlier for earlier, later in zip(decision_times_ms, decision_times_ms[1:])):
        raise ValueError("decision times must be monotonic")
    engine = CausalMicrostructureFeatureEngine(
        windows_seconds=tuple(policy["feature_windows_seconds"])
    )
    event_iterator = iter(events)
    next_event = next(event_iterator, None)
    decisions: list[dict[str, object]] = []
    for decision_time_ms in decision_times_ms:
        while next_event is not None and _event_time_ms(next_event) <= decision_time_ms:
            engine.consume(next_event)
            next_event = next(event_iterator, None)
        snapshot = engine.snapshot(decision_time_ms)
        feature_vector = feature_vector_for_side(
            snapshot.values,
            normalized_side,
            POLICY_FEATURE_COLUMNS,
        )
        would_enter, reasons = _would_enter(policy, feature_vector)
        if not would_enter and not include_non_entries:
            continue
        decisions.append(
            {
                "decision_time": datetime.fromtimestamp(decision_time_ms / 1000, tz=UTC).isoformat(),
                "decision_time_ms": decision_time_ms,
                "side": normalized_side,
                "horizon_seconds": int(policy["horizon_seconds"]),
                "probability": 1.0 if would_enter else 0.0,
                "threshold": 1.0,
                "would_enter": would_enter,
                "decision_reasons": reasons,
                "policy_sha256": _sha256(policy),
                "feature_vector_sha256": _sha256(feature_vector),
                "feature_vector": feature_vector,
            }
        )
    return {
        "scored_rows": len(decision_times_ms),
        "selected_count": sum(bool(decision["would_enter"]) for decision in decisions),
        "decisions": decisions,
    }


def _policy_from_report(report: dict[str, Any]) -> dict[str, Any]:
    _validate_report_boundary(report)
    config = report["config"]
    horizons = tuple(int(value) for value in config["horizons_seconds"])
    if len(horizons) != 1:
        raise ValueError("policy report must have exactly one horizon")
    return {
        "name": "spot_age_fixed_gate_v1",
        "source_experiment_stage": report["experiment_stage"],
        "horizon_seconds": horizons[0],
        "threshold_bps": float(config["expected_cost_bps"]),
        "initial_stop_bps": float(config["initial_stop_bps"]),
        "feature_windows_seconds": tuple(int(value) for value in config["feature_windows_seconds"]),
        "max_book_staleness_ms": int(config["max_book_staleness_ms"]),
        "base_thresholds": report["base_thresholds"],
        "frozen_gate": report["frozen_gate"],
        "side": "SELL",
        "order_submission_allowed": False,
        "execution_authorization": "none",
    }


def _validate_report_boundary(report: dict[str, Any]) -> None:
    if report.get("research_only") is not True:
        raise ValueError("policy report must be research_only")
    if report.get("order_submission_allowed") is not False:
        raise ValueError("policy report must not allow orders")
    if report.get("execution_authorization") != "none":
        raise ValueError("policy report must have execution_authorization=none")
    if report.get("approved_strategy") is not False:
        raise ValueError("policy report must not be an approved strategy")
    if "base_thresholds" not in report or "frozen_gate" not in report:
        raise ValueError("policy report must contain frozen thresholds")


def _would_enter(
    policy: dict[str, Any],
    feature_vector: dict[str, float],
) -> tuple[bool, list[str]]:
    reasons = []
    preflight_failed = False
    if feature_vector["book_available"] < 1:
        reasons.append("book_unavailable")
        preflight_failed = True
    if feature_vector["book_update_age_ms"] > float(policy["max_book_staleness_ms"]):
        reasons.append("book_stale")
        preflight_failed = True
    if feature_vector["spot_available"] < 1:
        reasons.append("spot_unavailable")
        preflight_failed = True
    if preflight_failed:
        return False, reasons
    for threshold in policy["base_thresholds"]:
        feature = str(threshold["feature"])
        direction = str(threshold["direction"])
        value = float(feature_vector[feature])
        cutoff = float(threshold["threshold"])
        if direction == "high":
            passed = value >= cutoff
        elif direction == "low":
            passed = value <= cutoff
        else:
            raise ValueError("base threshold direction must be high or low")
        reasons.append(
            f"base:{feature}:{direction}:{value:.12g}:{cutoff:.12g}:{'pass' if passed else 'fail'}"
        )
        if not passed:
            return False, reasons
    gate = policy["frozen_gate"]
    gate_feature = str(gate["feature"])
    gate_direction = str(gate["direction"])
    gate_value = float(feature_vector[gate_feature])
    gate_cutoff = float(gate["threshold"])
    if gate_direction == "high":
        gate_passed = gate_value >= gate_cutoff
    elif gate_direction == "low":
        gate_passed = gate_value <= gate_cutoff
    else:
        raise ValueError("gate direction must be high or low")
    reasons.append(
        "gate:"
        f"{gate_feature}:{gate_direction}:{gate_value:.12g}:{gate_cutoff:.12g}:"
        f"{'pass' if gate_passed else 'fail'}"
    )
    return gate_passed, reasons


def _event_time_ms(event: MicrostructureEvent) -> int:
    return int(event.event_time.timestamp() * 1000)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("datetime must include timezone")
    return parsed.astimezone(UTC)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_report(output_root: Path, report: dict[str, object]) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "report.json"
    payload = {
        **report,
        "report_path": str(output),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _console_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        "research_only": report["research_only"],
        "order_submission_allowed": report["order_submission_allowed"],
        "execution_authorization": report["execution_authorization"],
        "committed": report["committed"],
        "dry_run": report["dry_run"],
        "policy_sha256": report["policy_sha256"],
        "decision_count": report["decision_count"],
        "selected_count": report["selected_count"],
        "report_path": report["report_path"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
