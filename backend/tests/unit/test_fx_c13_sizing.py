"""The money-rules simulation: a cap per currency and direction, and a daily brake on new entries."""

import numpy as np
import pytest

from scripts.research import fx_c13_sizing as sizing


def run(net, entry, exit_, day, asset_a, sign_a, asset_b, sign_b, currency_cap=np.inf, brake=np.inf):
    n = len(net)
    return sizing.simulate(np.array(entry, float), np.array(exit_, float), np.array(net, float), np.ones(n),
                           np.ones(n), np.ones(n), np.full(n, 0.01), np.array(day), np.array(asset_a),
                           np.array(sign_a, float), np.array(asset_b), np.array(sign_b, float), 3, 1.0, currency_cap,
                           brake, 1000.0)


def test_two_bets_on_the_same_currency_hit_its_cap_and_an_opposite_one_does_not() -> None:
    # long EURUSD (EUR 0, USD 1), long GBPUSD (GBP 2, USD 1): both short USD; then short GBPUSD: long USD
    out = run([1.0, 1.0, 1.0], [0, 1, 2], [10, 10, 10], [0, 0, 0], [0, 2, 2], [1, 1, -1], [1, 1, 1], [-1, -1, 1],
              currency_cap=0.015)
    final, _, _, taken, _, _, currency, _ = out
    assert (taken, currency) == (2, 1) and final == pytest.approx(1020.0)


def test_the_daily_brake_stops_new_entries_until_the_next_day() -> None:
    # day 0: a loss of 3R (US$ 30 = 3%) realized, then two more entries that day, then one the next day
    out = run([-3.0, 1.0, 1.0, 1.0], [0, 5, 6, 20], [4, 8, 9, 25], [0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1],
              [-1, -1, -1, -1], [0, 0, 0, 0], brake=0.02)
    final, _, _, taken, _, _, _, braked = out
    assert (taken, braked) == (2, 2) and final == pytest.approx(1000.0 - 30.0 + 10.0)
