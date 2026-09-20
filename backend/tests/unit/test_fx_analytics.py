"""Bot performance numbers, checked by hand."""

from datetime import UTC, datetime

import pytest

from app.fx import analytics as an
from app.fx.analytics import ClosedTrade


def at(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp()


def trade(entry: str, exit_: str, pnl: float, r: float | None = None, side: int = 1) -> ClosedTrade:
    return ClosedTrade("EURUSD", side, at(entry), at(exit_), pnl, r)


TRADES = [
    trade("2024-05-13 08:00", "2024-05-13 08:30", +2.0, 1.6),
    trade("2024-05-13 09:00", "2024-05-13 09:10", -1.0, -1.0),
    trade("2024-05-14 15:00", "2024-05-14 15:55", -1.0, -1.0),
    trade("2024-05-14 09:00", "2024-05-14 09:20", +3.0, 2.4),
]


def test_the_headline_numbers() -> None:
    result = an.performance(TRADES, start_equity=500.0)

    assert (result.trades, result.wins, result.losses) == (4, 2, 2)
    assert result.win_rate == 0.5 and result.total_pnl == pytest.approx(3.0)
    assert result.return_pct == pytest.approx(0.6)
    assert result.profit_factor == pytest.approx(5.0 / 2.0)
    assert result.avg_r == pytest.approx((1.6 - 1.0 - 1.0 + 2.4) / 4)


def test_the_drawdown_is_peak_to_trough_of_the_equity_curve() -> None:
    # In order of exit: +2, -1, +3 (the 09:20 trade), -1: the curve is 502, 501, 504, 503.
    assert an.max_drawdown_pct([+2.0, -1.0, +3.0, -1.0], 500.0) == pytest.approx(100 * 1.0 / 502.0)
    assert an.max_drawdown_pct([-5.0, +10.0], 500.0) == pytest.approx(1.0)
    assert an.max_drawdown_pct([], 500.0) == 0.0
    with pytest.raises(ValueError):
        an.max_drawdown_pct([1.0], 0.0)


def test_time_in_position_counts_overlaps_once() -> None:
    overlapping = [trade("2024-05-13 08:00", "2024-05-13 09:00", 1.0), trade("2024-05-13 08:30", "2024-05-13 09:30", 1.0)]

    result = an.performance(overlapping, start_equity=500.0)

    assert result.seconds_in_position == 90 * 60
    assert result.share_of_time_in_position == pytest.approx(1.0)  # in a position for the whole span
    assert result.avg_hold_seconds == 60 * 60


def test_where_the_bot_wins_and_loses_by_london_hour_and_weekday() -> None:
    result = an.performance(TRADES, start_equity=500.0)

    # May is summer time: 08:00 UTC is 09:00 in London, 09:00 UTC is 10:00, 15:00 UTC is 16:00.
    assert result.by_hour_london[9] == an.Slice(1, pytest.approx(2.0))
    assert result.by_hour_london[10] == an.Slice(2, pytest.approx(2.0))  # the -1 and the +3
    assert result.by_hour_london[16].pnl == pytest.approx(-1.0)  # the 15:00 UTC trade is 16:00 London
    assert result.by_weekday[0].pnl == pytest.approx(1.0) and result.by_weekday[1].pnl == pytest.approx(2.0)


def test_best_and_worst_trades_and_the_no_loss_case() -> None:
    result = an.performance(TRADES, start_equity=500.0, top=1)

    assert result.best[0].pnl == 3.0 and result.worst[0].pnl == -1.0
    only_wins = an.performance([trade("2024-05-13 08:00", "2024-05-13 08:30", 1.0)], start_equity=500.0)
    assert only_wins.profit_factor is None


def test_an_empty_history_is_all_zeros_not_an_error() -> None:
    result = an.performance([], start_equity=500.0)

    assert result.trades == 0 and result.win_rate == 0.0 and result.max_drawdown_pct == 0.0
    assert result.avg_r is None and result.share_of_time_in_position == 0.0
