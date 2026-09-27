"""The sweep's money simulation: sizing at entry, results realized at exit, the two policies for a minimum volume that
risks more than the budget, the cap on the stops open at once, and compounding on or off."""

import numpy as np
import pytest

from scripts.research import fx_c13_sweep as sweep


def run(net, per_unit, risk=0.01, force=False, heat=1.0, compound=True, entry=(0.0, 5.0), exit_=(10.0, 20.0)):
    n = len(net)
    final, worst, worst_open, taken, small, capped, most, broke = sweep.simulate(
        np.array(entry[:n]), np.array(exit_[:n]), np.array(net, dtype=float), np.full(n, per_unit), np.ones(n),
        np.ones(n), risk, force, heat, compound, 1000.0)
    return {"final": final, "worst": worst, "worst_open": worst_open, "taken": taken, "small": small,
            "capped": capped, "most": most, "broke": broke}


def test_overlapping_trades_are_sized_on_the_realized_balance() -> None:
    out = run([2.0, -1.0], per_unit=1.0)
    # both enter on US$ 1,000 (the first has not exited when the second enters): 10 units, US$ 10 per R each
    assert out["final"] == pytest.approx(1010.0)
    assert out["worst"] == pytest.approx(10.0 / 1020.0)
    assert out["worst_open"] == pytest.approx(20.0 / 1000.0)  # both stops open at once
    assert (out["taken"], out["most"], out["broke"]) == (2, 2, False)


def test_a_later_entry_uses_the_balance_after_earlier_exits_only_when_compounding() -> None:
    later = {"entry": (0.0, 15.0), "exit_": (10.0, 20.0)}
    assert run([2.0, 1.0], 1.0, **later)["final"] == pytest.approx(1020.0 + 10.0)  # floor(0.01 * 1020) = 10 units
    assert run([2.0, 1.0], 0.1, **later)["final"] == pytest.approx(1020.0 + 10.2)  # 102 units of US$ 0.10
    assert run([2.0, 1.0], 0.1, compound=False, **later)["final"] == pytest.approx(1020.0 + 10.0)  # 100 units


def test_minimum_volume_above_budget_is_skipped_or_forced() -> None:
    skipped = run([1.0], per_unit=100.0)
    assert (skipped["taken"], skipped["small"]) == (0, 1) and skipped["final"] == pytest.approx(1000.0)
    forced = run([1.0], per_unit=100.0, force=True)
    assert forced["taken"] == 1 and forced["final"] == pytest.approx(1100.0)  # the minimum risks US$ 100, not US$ 10


def test_the_heat_cap_skips_what_would_pass_it() -> None:
    out = run([2.0, -1.0], per_unit=1.0, heat=0.015)  # US$ 10 open, another US$ 10 would make 2%
    assert (out["taken"], out["capped"]) == (1, 1) and out["final"] == pytest.approx(1020.0)


def test_the_grid_holds_the_frozen_management() -> None:
    assert list(sweep.GRID[sweep.FROZEN_G]) == [1.0, 0.0, 0.0, 3.0, 0.0, 0.0]
