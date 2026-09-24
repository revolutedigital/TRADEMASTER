"""Refresh live partial data and run a Fast185 observation snapshot.

This is a convenience runner for the Fast185 prospectively preregistered
candidate. It refreshes BTCUSDT/ETHUSDT/SOLUSDT diagnostic live partial
aggTrade partitions, reruns Fast177 to regenerate the source prospective rows,
then reruns Fast185 and appends a research-only snapshot event.

No credentials are used. No orders or ledgers are touched.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from scripts.research.fetch_binance_live_aggtrades_partition import (  # noqa: E402
    FUTURES_MARKET_URL,
    extend_and_normalize_live_aggtrades_partition,
    fetch_and_normalize_live_aggtrades_partition,
    parse_utc_datetime,
)
from scripts.research.run_fast174_crypto_cross_asset_leadlag_smoke import (  # noqa: E402
    DEFAULT_BTC_TRADE_ROOT,
    DEFAULT_ETH_TRADE_ROOT,
    DEFAULT_SOL_TRADE_ROOT,
)
from scripts.research.run_fast177_fast176_relaxed_edge_observation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as FAST177_OUTPUT_ROOT,
    run_fast177_fast176_relaxed_edge_observation,
)
from scripts.research.run_fast185_fast177_edge7p5_micro5_session_observation import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT as FAST185_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH as FAST185_REGISTRY_PATH,
    run_fast185_observation,
)


SYMBOL_ROOTS = {
    "BTCUSDT": DEFAULT_BTC_TRADE_ROOT.parents[1],
    "ETHUSDT": DEFAULT_ETH_TRADE_ROOT.parents[1],
    "SOLUSDT": DEFAULT_SOL_TRADE_ROOT.parents[1],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utc-date", type=date.fromisoformat, required=True)
    parser.add_argument("--start-at", type=parse_utc_datetime, required=True)
    parser.add_argument("--end-at", type=parse_utc_datetime, required=True)
    parser.add_argument("--market-base-url", default=FUTURES_MARKET_URL)
    parser.add_argument("--fast185-registry-path", type=Path, default=FAST185_REGISTRY_PATH)
    parser.add_argument("--replace-partial", action="store_true")
    parser.add_argument(
        "--full-replace",
        action="store_true",
        help="Fetch the full requested window instead of extending the existing live partial partition.",
    )
    args = parser.parse_args(argv)
    result = asyncio.run(
        run_fast185_live_partial_refresh_snapshot(
            utc_date=args.utc_date,
            start_at=args.start_at,
            end_at=args.end_at,
            market_base_url=args.market_base_url,
            fast185_registry_path=args.fast185_registry_path,
            replace_partial=args.replace_partial,
            incremental=not args.full_replace,
        )
    )
    print(json.dumps(console_summary(result), indent=2, sort_keys=True))  # noqa: T201
    return 0


async def run_fast185_live_partial_refresh_snapshot(
    *,
    utc_date: date,
    start_at: datetime,
    end_at: datetime,
    market_base_url: str = FUTURES_MARKET_URL,
    fast185_registry_path: Path = FAST185_REGISTRY_PATH,
    replace_partial: bool = False,
    incremental: bool = True,
) -> dict[str, Any]:
    start_at = start_at.astimezone(UTC)
    end_at = end_at.astimezone(UTC)
    manifests = {}
    for symbol, root in SYMBOL_ROOTS.items():
        refresh = (
            extend_and_normalize_live_aggtrades_partition
            if incremental
            else fetch_and_normalize_live_aggtrades_partition
        )
        manifest = await refresh(
            symbol=symbol,
            utc_date=utc_date,
            start_at=start_at,
            end_at=end_at,
            root=root,
            market_base_url=market_base_url,
            replace_partial=replace_partial,
        )
        manifests[symbol] = manifest.model_dump(mode="json")

    fast177_report = run_fast177_fast176_relaxed_edge_observation(
        trade_roots={
            "BTCUSDT": DEFAULT_BTC_TRADE_ROOT,
            "ETHUSDT": DEFAULT_ETH_TRADE_ROOT,
            "SOLUSDT": DEFAULT_SOL_TRADE_ROOT,
        },
        output_root=FAST177_OUTPUT_ROOT,
    )
    fast185_report = run_fast185_observation(output_root=FAST185_OUTPUT_ROOT)
    registry_event = build_fast185_snapshot_event(
        fast177_report=fast177_report,
        fast185_report=fast185_report,
        manifests=manifests,
        start_at=start_at,
        end_at=end_at,
    )
    append_jsonl(fast185_registry_path, registry_event)
    return {
        "manifests": manifests,
        "fast177_report": fast177_report,
        "fast185_report": fast185_report,
        "fast185_registry_event": registry_event,
    }


def build_fast185_snapshot_event(
    *,
    fast177_report: dict[str, Any],
    fast185_report: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    start_at: datetime,
    end_at: datetime,
) -> dict[str, Any]:
    summary = fast185_report["summary"]
    observation_summary = summary["observation_summary"]
    latest = latest_prospective_row(fast185_report)
    return {
        "event": "fast185_snapshot_extension",
        "date": end_at.astimezone(UTC).date().isoformat(),
        "research_only": True,
        "approved_strategy": False,
        "order_submission_allowed": False,
        "execution_authorization": "none",
        "orders_sent": False,
        "dry_run": True,
        "live_partial_source_mode": "binance_fapi_diagnostic_live_partial",
        "live_partial_window_start": start_at.astimezone(UTC).isoformat(),
        "live_partial_window_end": end_at.astimezone(UTC).isoformat(),
        "normalized_sha256_by_symbol": {
            symbol: str(manifest["normalized_sha256"]) for symbol, manifest in sorted(manifests.items())
        },
        "row_count_by_symbol": {
            symbol: int(manifest["row_count"]) for symbol, manifest in sorted(manifests.items())
        },
        "fast177_report_sha256": fast177_report["report_sha256"],
        "fast177_summary_sha256": fast177_report["artifacts"]["summary"]["sha256"],
        "fast177_prospective_diagnostic_rows_sha256": fast177_report["artifacts"][
            "prospective_diagnostic_rows"
        ]["sha256"],
        "status": fast185_report["status"],
        "status_reasons": fast185_report["status_reasons"],
        "prospective_row_count": int(summary["prospective_row_count"]),
        "eligible_observation_count": int(summary["eligible_observation_count"]),
        "observation_summary": {
            "decision_count": int(observation_summary["decision_count"]),
            "day_count": int(observation_summary["day_count"]),
            "stress_net_total_bps": float(observation_summary["stress_net_total_bps"]),
            "stress_net_mean_bps": float(observation_summary["stress_net_mean_bps"]),
            "positive_day_share": float(observation_summary["positive_day_share"]),
        },
        "prospective_latest_decision_time": None if latest is None else latest["decision_time"],
        "prospective_latest_edge_bps": None if latest is None else latest["edge"],
        "prospective_latest_regime_gate_value": None if latest is None else latest["regime_gate_value"],
        "prospective_latest_decision_hour_utc": None if latest is None else latest["fast185_decision_hour_utc"],
        "report_sha256": fast185_report["report_sha256"],
        "summary_sha256": fast185_report["artifacts"]["summary"]["sha256"],
        "prospective_diagnostics_sha256": fast185_report["artifacts"]["prospective_diagnostics"]["sha256"],
        "observations_sha256": fast185_report["artifacts"]["observations"]["sha256"],
    }


def latest_prospective_row(fast185_report: dict[str, Any]) -> dict[str, Any] | None:
    prospective_path = Path(fast185_report["artifacts"]["prospective_diagnostics"]["path"])
    if not prospective_path.exists():
        return None
    latest = None
    for line in prospective_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            latest = json.loads(line)
    return latest


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def console_summary(result: dict[str, Any]) -> dict[str, Any]:
    event = result["fast185_registry_event"]
    return {
        "research_only": event["research_only"],
        "approved_strategy": event["approved_strategy"],
        "order_submission_allowed": event["order_submission_allowed"],
        "execution_authorization": event["execution_authorization"],
        "orders_sent": event["orders_sent"],
        "live_partial_window_end": event["live_partial_window_end"],
        "status": event["status"],
        "status_reasons": event["status_reasons"],
        "prospective_row_count": event["prospective_row_count"],
        "eligible_observation_count": event["eligible_observation_count"],
        "observation_summary": event["observation_summary"],
        "report_sha256": event["report_sha256"],
        "summary_sha256": event["summary_sha256"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
