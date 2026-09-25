"""C6: the banca's context never looks at a bar or a trade result that is not closed at the entry."""

import numpy as np

from scripts.research import fx_c6_lab as c6


def test_the_last_closed_bar_is_strictly_before_the_entry_minute() -> None:
    last = np.array([9, 19, 29])  # bars end at minutes 9, 19 and 29

    np.testing.assert_array_equal(c6.last_closed(last, np.array([5, 10, 19, 20, 35])), [-1, 0, 0, 1, 2])


def test_recent_results_only_count_trades_closed_before_the_entry() -> None:
    entries = np.array([0.0, 10.0, 20.0, 30.0])
    exits = np.array([15.0, 25.0, 26.0, 40.0])
    net = np.array([1.0, 2.0, 4.0, 8.0])

    recent = c6.recent_mean(entries, exits, net, window=2)

    # at 0 and 10 nothing has closed; at 20 only the first (1.0); at 30 the last two closed (2.0 and 4.0)
    np.testing.assert_allclose(recent, [np.nan, np.nan, 1.0, 3.0])


def test_every_instrument_has_a_class() -> None:
    from scripts.research import fx_c3_lab as c3

    classes = {c6.instrument_class(s) for s in c3.GROUP_A + c3.GROUP_B}
    assert classes == {"major", "cross", "metal", "index", "energy", "dollar_index"}
