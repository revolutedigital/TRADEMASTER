"""C8: decision points only while the trade is in profit, and the banked result is the next open."""

import numpy as np
import pytest

from scripts.research import fx_c8_exit_banca as c8


def test_decisions_only_in_profit_and_banking_uses_the_next_open() -> None:
    n = 130  # enough daily bars for the indicators to exist (they only become features)
    close = np.full(n, 1.00)
    close[120:] = [1.00, 1.02, 0.99, 1.05, 1.04, 1.03, 1.00, 1.00, 1.00, 1.00]
    opens = np.concatenate(([1.0], close[:-1]))
    d1 = np.column_stack([opens, np.maximum(opens, close) + 0.001, np.minimum(opens, close) - 0.001, close])
    d1_last = np.arange(n) * 10 + 9  # ten minutes per bar
    # long at minute 1205 (inside bar 120) at 1.00, exits at minute 1265 (inside bar 126) with net 0
    trades = np.array([[1, 1205, 1265, 1.00, 1.00, 0.02, 3, 0.0, -0.01]])

    rows = c8.decision_points(trades, d1_last, d1, cost=0.0)

    assert [r["bar"] for r in rows] == [121, 123, 124, 125]  # bar 122 closes at a loss; 120 and 126 are not in profit / exit
    first = rows[0]
    assert first["profit"] == pytest.approx(1.0)  # (1.02 - 1.00) / 0.02
    assert first["realized"] == pytest.approx((1.02 - 1.00) / 0.02)  # the next open is the 1.02 close
    assert first["remaining"] == pytest.approx(0.0 - first["realized"])
