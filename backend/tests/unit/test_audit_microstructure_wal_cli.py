"""CLI range selection for the prospective WAL auditor."""

from __future__ import annotations

from datetime import date

import pytest

from scripts.research.audit_microstructure_wal import _resolve_range


def test_resolve_range_defaults_to_previous_complete_utc_day() -> None:
    start_date, end_date = _resolve_range(
        None,
        None,
        None,
        today=date(2026, 3, 2),
    )

    assert start_date == date(2026, 3, 1)
    assert end_date == date(2026, 3, 1)


def test_resolve_range_supports_sixty_day_rolling_window() -> None:
    start_date, end_date = _resolve_range(
        None,
        None,
        date(2026, 3, 1),
        rolling_days=60,
    )

    assert start_date == date(2026, 1, 1)
    assert end_date == date(2026, 3, 1)


def test_resolve_range_rejects_ambiguous_rolling_start() -> None:
    with pytest.raises(SystemExit, match="combined only"):
        _resolve_range(
            None,
            date(2026, 1, 1),
            date(2026, 3, 1),
            rolling_days=60,
        )


def test_resolve_range_rejects_non_positive_rolling_days() -> None:
    with pytest.raises(SystemExit, match="positive"):
        _resolve_range(None, None, date(2026, 3, 1), rolling_days=0)
