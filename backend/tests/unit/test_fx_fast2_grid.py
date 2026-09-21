"""The round-2 grid: its size is fixed, round 1's configurations are inside it, and the file cannot drift."""

import numpy as np
import pytest

from app.fx.strategies import fixing_flow, pairs_spread, session_breakout, spike_fade
from scripts.research import fx_fast2_grid as grid
from scripts.research import fx_fast_lab as v1


def test_the_grid_has_212_unique_combinations_by_family() -> None:
    points = grid.grid_points()

    assert len(points) == 212 and len({p["key"] for p in points}) == 212
    counts: dict[str, int] = {}
    for point in points:
        counts[point["family"]] = counts.get(point["family"], 0) + 1
    assert counts == {"F1a": 48, "F1b": 48, "F2a": 18, "F2b": 18, "F3": 32, "F5a": 24, "F5b": 24}


def test_every_combination_builds_valid_parameters_for_each_of_its_pairs() -> None:
    configs = grid.build_configurations(grid.grid_points())

    for config in configs.values():
        for pair in config.pairs:
            params = config.params(pair)
            assert params.dtype == np.float64 and np.isfinite(params).all()


@pytest.mark.parametrize("v1_key,grid_key", [
    ("F1a", "F1a.rr1.5.w0.3-1.2.se11:00.x16:30"), ("F1b", "F1b.rr1.5.w0.3-1.2.se11:00.x16:00"),
    ("F2a", "F2a.in15:00.out15:55.atr3.0"), ("F2b", "F2b.in16:05.out17:00.atr3.0"),
    ("F3a", "F3.th4.0.stop0.5.ret0.5.hold12"), ("F3b", "F3.th6.0.stop0.5.ret0.5.hold12"),
    ("F5a", "F5a.z2.0.stop3.5.hold120"), ("F5b", "F5b.z2.0.stop3.5.hold120"),
])
def test_the_round_1_configurations_are_inside_the_grid_with_identical_parameters(v1_key: str, grid_key: str) -> None:
    old, new = v1.configurations()[v1_key], grid.build_configurations(grid.grid_points())[grid_key]

    assert (old.seconds, old.pairs, old.state_size) == (new.seconds, new.pairs, new.state_size)
    for pair in old.pairs:
        np.testing.assert_array_equal(old.params(pair), new.params(pair))


def test_the_grid_file_is_the_generated_grid(tmp_path) -> None:
    path = tmp_path / "grid.json"
    grid.write_grid(path)

    assert grid.load_grid(path) == grid.grid_points()
    if grid.DEFAULT_GRID.exists():
        assert grid.DEFAULT_GRID.read_text() == path.read_text()  # the committed file is what the code generates


def test_the_step_functions_are_the_lab_ones() -> None:
    configs = grid.build_configurations(grid.grid_points())

    assert configs["F3.th4.0.stop0.5.ret0.5.hold12"].step is spike_fade.spike_fade_step
    assert configs["F1a.rr1.5.w0.3-1.2.se11:00.x16:30"].step is session_breakout.session_breakout_step
    assert configs["F2a.in15:00.out15:55.atr3.0"].step is fixing_flow.fixing_flow_step
    assert configs["F5a.z2.0.stop3.5.hold120"].step is pairs_spread.pairs_spread_step
