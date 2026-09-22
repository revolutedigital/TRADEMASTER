"""Score frozen top-p shadow entries from research parquet partitions.

Default mode is a dry-run: it scores the frozen artifact and prints the entry
count, but does not touch the database. Use --commit only after the prospective
shadow partition has been explicitly opened in the research ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.models.base import async_session_factory
from app.services.research.shadow_policy_runner import (
    record_frozen_top_p_shadow_batch,
    score_frozen_top_p_shadow_frame,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--policy-artifact", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to atomically write the scored batch shadow report.",
    )
    parser.add_argument(
        "--include-non-entries",
        action="store_true",
        help="Record every scored decision, not just probability >= top-p threshold.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Append selected shadow signals to the research ledger.",
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be positive")
    artifact = json.loads(arguments.policy_artifact.read_text(encoding="utf-8"))
    frame = _load_frame(
        arguments.dataset_root,
        start_date=arguments.start_date,
        end_date=arguments.end_date,
    )
    if arguments.commit:
        result = asyncio.run(
            _commit_batch(
                experiment_id=arguments.experiment_id,
                artifact=artifact,
                frame=frame,
                include_non_entries=arguments.include_non_entries,
                limit=arguments.limit,
            )
        )
        if arguments.output is not None:
            _write_json_report(arguments.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))  # noqa: T201
        return 0

    selection = score_frozen_top_p_shadow_frame(
        frame,
        artifact=artifact,
        include_non_entries=arguments.include_non_entries,
    )
    decisions = selection.decisions[: arguments.limit] if arguments.limit else selection.decisions
    report = {
        **asdict(selection),
        "selected_count_after_limit": len(decisions),
        "dry_run": True,
        "commit_required_to_write_ledger": True,
    }
    report["decisions"] = [
        {
            "decision_time": decision.decision_time.isoformat(),
            "side": decision.side,
            "horizon_seconds": decision.horizon_seconds,
            "probability": decision.probability,
            "threshold": decision.threshold,
            "would_enter": decision.would_enter,
            "model_sha256": decision.model_sha256,
            "feature_vector_sha256": decision.feature_vector_sha256,
        }
        for decision in decisions
    ]
    if arguments.output is not None:
        _write_json_report(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))  # noqa: T201
    return 0


async def _commit_batch(
    *,
    experiment_id: str,
    artifact: dict[str, object],
    frame: pd.DataFrame,
    include_non_entries: bool,
    limit: int | None,
) -> dict[str, object]:
    async with async_session_factory() as session:
        try:
            result = await record_frozen_top_p_shadow_batch(
                session,
                experiment_id=experiment_id,
                artifact=artifact,
                frame=frame,
                include_non_entries=include_non_entries,
                limit=limit,
            )
            await session.commit()
            return {
                **asdict(result),
                "committed": True,
                "research_only": True,
            }
        except Exception:
            await session.rollback()
            raise


def _load_frame(
    dataset_root: Path,
    *,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    paths = [
        path
        for path in sorted(dataset_root.glob("date=*/research_rows.parquet"))
        if _partition_in_range(path, start_date=start_date, end_date=end_date)
    ]
    if not paths:
        raise FileNotFoundError("no research dataset partitions matched the requested dates")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _partition_in_range(path: Path, *, start_date: date | None, end_date: date | None) -> bool:
    partition_date = date.fromisoformat(path.parent.name.removeprefix("date="))
    if start_date is not None and partition_date < start_date:
        return False
    if end_date is not None and partition_date > end_date:
        return False
    return True


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
