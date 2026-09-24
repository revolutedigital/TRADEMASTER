from __future__ import annotations

from datetime import date

from scripts.research.run_fast166_fast151_dukascopy_forward_probe import (
    TARGET_PAIRS,
    fast166_status,
    summarize_dates,
)


def _row(pair: str, probe_date: date, *, rows: int, post: int = 0, last: str | None = None) -> dict:
    return {
        "pair": pair,
        "date": probe_date.isoformat(),
        "row_count": rows,
        "last": last,
        "has_post_preregistration_minutes": post > 0,
        "post_preregistration_minutes": post,
        "cache_bytes": {"BID": 10 if rows else 0, "ASK": 10 if rows else 0},
    }


def test_fast166_status_detects_alive_source_without_preregistration_date() -> None:
    control = date(2026, 9, 23)
    prereg = date(2026, 9, 24)
    rows = [_row(pair, control, rows=100, last="2026-09-23T23:59:00+00:00") for pair in TARGET_PAIRS]
    rows += [_row(pair, prereg, rows=0) for pair in TARGET_PAIRS]

    status, reasons = fast166_status(rows=rows, preregistration_date=prereg, control_date=control)

    assert status == "FAST166_DUKASCOPY_SOURCE_ALIVE_BUT_PREREGISTRATION_DATE_UNAVAILABLE"
    assert "control_date_has_m1_data" in reasons
    assert "preregistration_date_has_no_m1_data" in reasons
    assert "not_approved_strategy" in reasons


def test_fast166_status_detects_all_target_pairs_with_post_preregistration_m1() -> None:
    control = date(2026, 9, 23)
    prereg = date(2026, 9, 24)
    rows = [_row(pair, control, rows=100) for pair in TARGET_PAIRS]
    rows += [_row(pair, prereg, rows=100, post=10) for pair in TARGET_PAIRS]

    status, reasons = fast166_status(rows=rows, preregistration_date=prereg, control_date=control)

    assert status == "FAST166_DUKASCOPY_HAS_POST_PREREGISTRATION_M1_IMPORT_REQUIRED"
    assert "all_target_pairs_have_post_preregistration_m1" in reasons
    assert "canonical_overlay_import_required" in reasons


def test_summarize_dates_reports_pair_coverage_and_cache_size() -> None:
    probe_date = date(2026, 9, 24)
    rows = [
        _row("AUDUSD", probe_date, rows=0),
        _row("EURUSD", probe_date, rows=100, post=5, last="2026-09-24T12:00:00+00:00"),
    ]

    summary = summarize_dates(rows)

    assert summary["2026-09-24"]["pairs"] == ["AUDUSD", "EURUSD"]
    assert summary["2026-09-24"]["total_rows"] == 100
    assert summary["2026-09-24"]["pairs_with_rows"] == ["EURUSD"]
    assert summary["2026-09-24"]["pairs_with_post_preregistration_minutes"] == ["EURUSD"]
    assert summary["2026-09-24"]["cache_bytes_total"] == 20
