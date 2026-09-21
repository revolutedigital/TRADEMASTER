"""Round 2 of the lab: the three samples keep their roles and the grid runs through round 1's machinery."""

import numpy as np

from scripts.research import fx_fast2_grid as grid
from scripts.research import fx_fast2_lab as lab2
from scripts.research import fx_fast_lab as lab
from tests.unit.test_fx_fast_lab import RATES, dataset

F1_LOW = "F1a.rr1.0.w0.3-1.2.se11:00.x16:30"
F1_HIGH = "F1a.rr2.5.w0.3-1.2.se11:00.x16:30"
SAMPLE_KEYS = [F1_LOW, "F2a.in15:00.out15:55.atr3.0", "F3.th3.0.stop0.5.ret0.5.hold12", "F5b.z2.0.stop3.5.hold120"]


def chosen(keys):
    everything = lab2.configurations()
    return {key: everything[key] for key in keys}


def test_the_three_samples_are_ordered_and_only_the_discovery_uses_the_older_data() -> None:
    months = [lab2.SAMPLES[stage][0] for stage in ("discovery", "replication", "confirmation")]
    directories = [lab2.SAMPLES[stage][1] for stage in ("discovery", "replication", "confirmation")]

    assert months[0][1] < months[1][0] and months[1][1] < months[2][0]
    assert directories[0] != directories[1] and directories[1] == directories[2]
    assert months[1] == lab.DISCOVERY and months[2] == lab.CONFIRMATION  # round 1's samples, unchanged


def test_the_lab_runs_exactly_the_committed_grid() -> None:
    assert set(lab2.configurations()) == {point["key"] for point in grid.load_grid()} and len(lab2.configurations()) == 212


def test_grid_configurations_of_every_family_run_through_the_lab() -> None:
    matrices, windows, universe = dataset()

    result = lab.run_sample(matrices, windows, universe, RATES, chosen(SAMPLE_KEYS))

    assert result.keys == SAMPLE_KEYS and result.sums.shape == (len(universe), len(SAMPLE_KEYS))
    assert np.isfinite(result.sums).all()


def test_sharing_prepared_bars_between_configurations_changes_nothing() -> None:
    matrices, windows, universe = dataset()

    together = lab.run_sample(matrices, windows, universe, RATES, chosen([F1_LOW, F1_HIGH]))
    alone = [lab.run_sample(matrices, windows, universe, RATES, chosen([key])) for key in (F1_LOW, F1_HIGH)]

    for column, single in enumerate(alone):
        np.testing.assert_array_equal(together.sums[:, column], single.sums[:, 0])
        np.testing.assert_array_equal(together.counts[:, column], single.counts[:, 0])
        assert together.stress_mean[column] == single.stress_mean[0] or np.isnan(single.stress_mean[0])
