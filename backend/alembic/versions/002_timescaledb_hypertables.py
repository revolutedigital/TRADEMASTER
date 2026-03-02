"""Enable TimescaleDB and convert OHLCV to hypertable.

Revision ID: 002_timescaledb
Revises: 001_initial
Create Date: 2026-02-28
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "002_timescaledb"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable TimescaleDB extension (safe if already enabled)
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    # Convert OHLCV table to hypertable (partitioned by open_time)
    op.execute(
        "SELECT create_hypertable('ohlcv', 'open_time', "
        "migrate_data => true, if_not_exists => true)"
    )

    # Enable compression on the hypertable
    op.execute(
        "ALTER TABLE ohlcv SET ("
        "timescaledb.compress, "
        "timescaledb.compress_segmentby = 'symbol,interval'"
        ")"
    )

    # Add compression policy: compress chunks older than 7 days
    op.execute(
        "SELECT add_compression_policy('ohlcv', INTERVAL '7 days', "
        "if_not_exists => true)"
    )

    # Add data retention policy: drop 1-minute data older than 90 days
    op.execute(
        "SELECT add_retention_policy('ohlcv', INTERVAL '90 days', "
        "if_not_exists => true)"
    )


def downgrade() -> None:
    # Remove policies
    op.execute("SELECT remove_retention_policy('ohlcv', if_exists => true)")
    op.execute("SELECT remove_compression_policy('ohlcv', if_exists => true)")
    # Note: cannot easily revert hypertable to regular table
