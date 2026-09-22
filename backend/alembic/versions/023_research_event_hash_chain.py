"""Add tamper-evident hash chain to research experiment events.

Revision ID: 023
Revises: 022
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op


revision: str = "023"
down_revision: str | None = "022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_experiment_events",
        sa.Column("previous_event_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "research_experiment_events",
        sa.Column("event_sha256", sa.String(length=64), nullable=True),
    )
    _backfill_event_chain()
    op.create_index(
        "ix_research_experiment_events_chain",
        "research_experiment_events",
        ["experiment_id", "previous_event_sha256"],
    )
    op.create_index(
        "ix_research_experiment_events_event_sha256",
        "research_experiment_events",
        ["event_sha256"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_experiment_events_event_sha256",
        table_name="research_experiment_events",
    )
    op.drop_index(
        "ix_research_experiment_events_chain",
        table_name="research_experiment_events",
    )
    op.drop_column("research_experiment_events", "event_sha256")
    op.drop_column("research_experiment_events", "previous_event_sha256")


def _backfill_event_chain() -> None:
    connection = op.get_bind()
    rows = connection.execute(
        sa.text(
            """
            SELECT id, experiment_id, kind, payload_json, occurred_at
            FROM research_experiment_events
            ORDER BY experiment_id, occurred_at, id
            """
        )
    ).mappings()
    latest_by_experiment: dict[str, str | None] = {}
    for row in rows:
        experiment_id = str(row["experiment_id"])
        previous_event_sha256 = latest_by_experiment.get(experiment_id)
        event_sha256 = _event_sha256(
            experiment_id=experiment_id,
            kind=str(row["kind"]),
            payload_json=str(row["payload_json"] or "{}"),
            occurred_at=row["occurred_at"],
            previous_event_sha256=previous_event_sha256,
        )
        connection.execute(
            sa.text(
                """
                UPDATE research_experiment_events
                SET previous_event_sha256 = :previous_event_sha256,
                    event_sha256 = :event_sha256
                WHERE id = :event_id
                """
            ),
            {
                "event_id": row["id"],
                "previous_event_sha256": previous_event_sha256,
                "event_sha256": event_sha256,
            },
        )
        latest_by_experiment[experiment_id] = event_sha256


def _event_sha256(
    *,
    experiment_id: str,
    kind: str,
    payload_json: str,
    occurred_at: object,
    previous_event_sha256: str | None,
) -> str:
    encoded = json.dumps(
        {
            "experiment_id": experiment_id,
            "kind": kind,
            "payload_json": payload_json,
            "occurred_at": _utc_iso(occurred_at),
            "previous_event_sha256": previous_event_sha256,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _utc_iso(value: object) -> str:
    if isinstance(value, datetime):
        observed = value
    elif isinstance(value, str):
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        observed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    return observed.astimezone(UTC).isoformat().replace("+00:00", "Z")
