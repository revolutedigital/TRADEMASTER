import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.research.run_fast60_fast57_shadow_dry_run import WalRootRef
from scripts.research.run_fast81_fast80_prospective_update import (
    audit_wal_roots_for_fast80,
    fast80_preregistration_datetime,
    run_fast81_fast80_prospective_update,
)


def test_fast81_reports_no_post_preregistration_wal_without_running_chain(
    tmp_path: Path,
) -> None:
    cutoff = fast80_preregistration_datetime()
    wal_root = tmp_path / "prospective-wal-before"
    write_trade_events(
        wal_root,
        cutoff - timedelta(minutes=20),
        cutoff - timedelta(minutes=10),
    )

    report = run_fast81_fast80_prospective_update(
        artifact_report=tmp_path / "missing-artifact-not-needed.json",
        wal_roots=(WalRootRef(label="before", path=wal_root),),
        output_root=tmp_path / "out",
    )

    assert report["status"] == "FAST81_NO_POST_PREREG_WAL_ROOTS"
    assert report["post_preregistration_wal_root_count"] == 0
    assert report["approved_strategy"] is False
    assert report["order_submission_allowed"] is False
    assert "fast60" not in report


def test_fast81_selects_wal_root_that_extends_after_preregistration(
    tmp_path: Path,
) -> None:
    cutoff = fast80_preregistration_datetime()
    before_root = tmp_path / "prospective-wal-before"
    overlapping_root = tmp_path / "prospective-wal-overlap"
    write_trade_events(
        before_root,
        cutoff - timedelta(minutes=30),
        cutoff - timedelta(minutes=20),
    )
    write_trade_events(
        overlapping_root,
        cutoff - timedelta(minutes=5),
        cutoff + timedelta(minutes=5),
    )

    audit = audit_wal_roots_for_fast80(
        wal_roots=(
            WalRootRef(label="before", path=before_root),
            WalRootRef(label="overlap", path=overlapping_root),
        ),
        dates=None,
        preregistration_timestamp=cutoff,
    )

    selected_labels = [item.root.label for item in audit if item.selected]
    statuses = {item.root.label: item.status for item in audit}

    assert selected_labels == ["overlap"]
    assert statuses["before"] == "FAST81_WAL_ENDS_BEFORE_PREREGISTRATION"
    assert statuses["overlap"] == "FAST81_WAL_OVERLAPS_POST_PREREGISTRATION"


def write_trade_events(
    wal_root: Path,
    first_event_time: datetime,
    second_event_time: datetime,
) -> None:
    path = wal_root / "trade" / f"date={first_event_time.date().isoformat()}" / "events.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event_time": normalize(first_event_time), "price": 1.0, "quantity": 1.0},
        {"event_time": normalize(second_event_time), "price": 1.0, "quantity": 1.0},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def normalize(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
