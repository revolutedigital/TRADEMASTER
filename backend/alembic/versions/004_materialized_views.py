"""Materialized views for performance analytics.

Revision ID: 004_materialized_views
Revises: 003_add_model_and_backtest_fields
Create Date: 2025-10-01 00:00:00.000000
"""

from alembic import op

revision = "004_materialized_views"
down_revision = "003_add_model_and_backtest_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Daily returns materialized view
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_returns AS
        SELECT
            symbol,
            date_trunc('day', open_time) AS day,
            FIRST_VALUE(open) OVER (PARTITION BY symbol, date_trunc('day', open_time) ORDER BY open_time) AS day_open,
            MAX(high) AS day_high,
            MIN(low) AS day_low,
            LAST_VALUE(close) OVER (PARTITION BY symbol, date_trunc('day', open_time) ORDER BY open_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS day_close,
            SUM(volume) AS day_volume,
            COUNT(*) AS candle_count
        FROM ohlcv
        WHERE interval = '1h'
        GROUP BY symbol, date_trunc('day', open_time), open_time, open, close, high, low, volume
    """)

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_returns_symbol_day
        ON mv_daily_returns (symbol, day)
    """)

    # Monthly P&L materialized view
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_monthly_pnl AS
        SELECT
            date_trunc('month', created_at) AS month,
            SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END) AS gross_profit,
            SUM(CASE WHEN realized_pnl < 0 THEN realized_pnl ELSE 0 END) AS gross_loss,
            SUM(realized_pnl) AS net_pnl,
            COUNT(*) AS trade_count,
            COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) AS winning_trades,
            COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) AS losing_trades,
            AVG(realized_pnl) AS avg_pnl_per_trade
        FROM trades
        WHERE status = 'FILLED' AND realized_pnl IS NOT NULL
        GROUP BY date_trunc('month', created_at)
    """)

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_monthly_pnl_month
        ON mv_monthly_pnl (month)
    """)

    # Hourly volume profile materialized view
    op.execute("""
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_volume_profile AS
        SELECT
            symbol,
            EXTRACT(HOUR FROM open_time) AS hour_of_day,
            AVG(volume) AS avg_volume,
            AVG(high - low) AS avg_range,
            COUNT(*) AS sample_count
        FROM ohlcv
        WHERE interval = '1h'
        GROUP BY symbol, EXTRACT(HOUR FROM open_time)
    """)

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_volume_profile_symbol_hour
        ON mv_volume_profile (symbol, hour_of_day)
    """)


def downgrade() -> None:
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_volume_profile CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_monthly_pnl CASCADE")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_daily_returns CASCADE")
