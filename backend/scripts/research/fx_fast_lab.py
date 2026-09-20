"""Lab of the fast strategies: runs the pre-registered configurations and applies the criterion.

Order enforced by the registry (`fx_fast_registry`): `calibrate` (300 placebo datasets, measures the
false-approval rate of the whole criterion), then `discover` (real discovery sample), then
`confirm` (frozen sample, only what discovery approved). See docs/forex/fast-preregistration.md.

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from app.fx import strategy as fx
from app.fx.bars import aggregate_minutes
from app.fx.instruments import STANDARD_LOT_UNITS, ConversionRates, Instrument, pip_value
from app.fx.sessions import fx_day
from app.fx.sim import core
from app.fx.sim.costs import FUSION_ZERO, STRESS, RolloverCalendar, commission_round_trip_pips
from app.fx.sim.costs import finalize_trades, prepare_run
from app.fx.strategies import controls, fixing_flow, pairs_spread, session_breakout, spike_fade
from scripts.research import fx_fast_placebo as placebo
from scripts.research import fx_fast_registry as registry
from scripts.research import fx_fast_stats as stats
from scripts.research.fx_dataset import ALL_PAIRS
from scripts.research.fx_fast_manifest import DEFAULT_MANIFEST, load_m1

DISCOVERY = ("2019-01", "2024-08")
CONFIRMATION = ("2024-09", "2026-08")
SYNTHETIC = "AUDNZD"
USD_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF")
PLACEBO_REPLICATES = 300
PLACEBO_SEED = 20260921
FALSE_APPROVAL_LIMIT = 0.08
MIN_LOT_STOP_PIPS = {"JPY": 18.75, "other": 12.5}
CALENDAR = RolloverCalendar()
LAB_DIR = Path("data/lab")
REPORT_DIR = DEFAULT_MANIFEST.parent


@dataclass(frozen=True)
class Configuration:
    key: str
    seconds: int
    pairs: tuple[str, ...]
    step: Callable
    init: Callable
    state_size: int
    params: Callable[[str], np.ndarray]


def configurations() -> dict[str, Configuration]:
    every = tuple(ALL_PAIRS)
    sb, ff, sf, ps, ct = session_breakout, fixing_flow, spike_fade, pairs_spread, controls

    def breakout(key, builder):
        return Configuration(key, 900, every, sb.session_breakout_step, sb.session_breakout_init,
                             sb.SESSION_BREAKOUT_STATE_SIZE, lambda pair: builder())

    def fixing(key, builder):
        return Configuration(key, 300, USD_PAIRS, ff.fixing_flow_step, ff.fixing_flow_init,
                             ff.FIXING_FLOW_STATE_SIZE, builder)

    def fade(key, builder):
        return Configuration(key, 300, every, sf.spike_fade_step, sf.spike_fade_init,
                             sf.SPIKE_FADE_STATE_SIZE, lambda pair: builder())

    def reversion(key, pair):
        return Configuration(key, 3600, (pair,), ps.pairs_spread_step, ps.pairs_spread_init,
                             ps.PAIRS_SPREAD_STATE_SIZE, lambda _: ps.pairs_spread_params())

    def pip(pair):
        return Instrument.from_symbol(pair).pip_size

    table = [
        breakout("F1a", sb.london_open_breakout_params), breakout("F1b", sb.new_york_open_breakout_params),
        fixing("F2a", ff.pre_fixing_params), fixing("F2b", ff.post_fixing_params),
        fade("F3a", sf.spike_fade_4_params), fade("F3b", sf.spike_fade_6_params),
        reversion("F5a", "EURGBP"), reversion("F5b", SYNTHETIC),
        Configuration("C1", 60, every, ct.bollinger_reversion_step, ct.bollinger_reversion_init,
                      ct.BOLLINGER_REVERSION_STATE_SIZE, lambda pair: ct.bollinger_reversion_params(pip(pair))),
        Configuration("C2", 300, every, ct.momentum_burst_step, ct.momentum_burst_init,
                      ct.MOMENTUM_BURST_STATE_SIZE, lambda pair: ct.momentum_burst_params(pip(pair))),
    ]
    return {config.key: config for config in table}


# --- data -------------------------------------------------------------------------------------


def prepare_matrices(data_dir: Path, out_dir: Path) -> None:
    """Convert the M1 parquet files to memory-mappable matrices, once, for all workers to share."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for pair in ALL_PAIRS:
        np.save(out_dir / f"{pair}.npy", fx.bars_to_matrix(load_m1(pair, data_dir)))


def open_matrices(directory: Path) -> dict[str, np.ndarray]:
    return {pair: np.load(directory / f"{pair}.npy", mmap_mode="r") for pair in ALL_PAIRS}


def included_windows(manifest: pd.DataFrame, pair: str, sample: tuple[str, str]) -> list[tuple[float, float]]:
    """Runs of consecutive included months of the sample, as (start, end) seconds since the epoch."""
    legs = ("AUDUSD", "NZDUSD") if pair == SYNTHETIC else (pair,)
    months = None
    for leg in legs:
        rows = manifest[(manifest["pair"] == leg) & manifest["included"]]["month"]
        found = {m for m in rows if sample[0] <= m <= sample[1]}
        months = found if months is None else months & found
    windows: list[tuple[pd.Period, pd.Period]] = []
    for month in sorted(pd.Period(m) for m in months or ()):
        if windows and windows[-1][1] + 1 == month:
            windows[-1] = (windows[-1][0], month)
        else:
            windows.append((month, month))
    seconds = lambda period: float(period.start_time.tz_localize("UTC").timestamp())  # noqa: E731
    return [(seconds(first), seconds(last + 1)) for first, last in windows]


@njit(cache=True)
def fx_day_array(seconds):
    out = np.empty(seconds.shape[0], dtype=np.int64)
    for i in range(seconds.shape[0]):
        out[i] = fx_day(seconds[i])
    return out


def window_slice(matrix: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    times = matrix[:, fx.BAR_TIME]
    low, high = np.searchsorted(times, window[0]), np.searchsorted(times, window[1])
    return matrix[low:high]


def day_universe(matrices: dict[str, np.ndarray], windows: dict[str, list]) -> np.ndarray:
    days = [np.unique(fx_day_array(window_slice(matrices[p], w)[:, fx.BAR_TIME]))
            for p in windows if p != SYNTHETIC for w in windows[p]]
    return np.unique(np.concatenate(days))


def median_rates(matrices: dict[str, np.ndarray]) -> ConversionRates:
    return ConversionRates({p: float(np.median(0.5 * (matrices[p][:, fx.BID_CLOSE] + matrices[p][:, fx.ASK_CLOSE])))
                            for p in USD_PAIRS})


# --- running configurations ---------------------------------------------------------------------


@dataclass
class SampleResult:
    keys: list[str]
    sums: np.ndarray
    counts: np.ndarray
    pair_means: dict[str, dict[str, float]]
    stress_mean: np.ndarray
    fits_min_lot: np.ndarray

    @property
    def trades(self) -> np.ndarray:
        return self.counts.sum(axis=0)


def commission_pips(pair: str, price: float, rates: ConversionRates, base_per_side: float) -> float:
    """Round-trip commission in pips; the broker charges `base_per_side` of the pair's base currency per lot."""
    if pair == SYNTHETIC:  # one lot of AUDUSD against `price` lots of NZDUSD, both ways
        instrument = Instrument.from_symbol(SYNTHETIC)
        per_side = base_per_side * (rates.usd_per("AUD") + price * rates.usd_per("NZD"))
        return 2 * per_side / pip_value(instrument, STANDARD_LOT_UNITS, rates)
    return commission_round_trip_pips(
        Instrument.from_symbol(pair), price, rates, base_currency_per_lot_per_side=base_per_side
    )


def pair_minutes(pair: str, matrices, window, coins) -> np.ndarray:
    """The M1 bars of a pair (or the synthetic cross) inside a window, placebo-randomised if asked."""
    def leg(name: str) -> np.ndarray:
        matrix = np.ascontiguousarray(window_slice(matrices[name], window))
        if coins is None:
            return matrix
        table, first = coins
        return placebo.sign_randomize(matrix, placebo.coins_for(matrix, table, first))

    if pair == SYNTHETIC:
        return pairs_spread.synthetic_cross(leg("AUDUSD"), leg("NZDUSD"))
    return leg(pair)


def run_sample(matrices, windows, universe, rates, configs, coins=None) -> SampleResult:
    keys = list(configs)
    index = {key: k for k, key in enumerate(keys)}
    days, values, config_ids, pairs_of, fits = [], [], [], [], []
    stress_sum, stress_count = np.zeros(len(keys)), np.zeros(len(keys))
    for pair, pair_windows in windows.items():
        instrument = Instrument.from_symbol(pair)
        for window in pair_windows:
            minutes = pair_minutes(pair, matrices, window, coins)
            if len(minutes) < 2:
                continue
            price = float(np.median(0.5 * (minutes[:, fx.BID_CLOSE] + minutes[:, fx.ASK_CLOSE])))
            for key, config in configs.items():
                if pair not in config.pairs:
                    continue
                bars = aggregate_minutes(minutes, config.seconds)
                params = config.params(pair)
                for scenario in (FUSION_ZERO, STRESS):
                    scenario_bars, slippage = prepare_run(bars, instrument, scenario)
                    if pair == SYNTHETIC:  # both legs slip
                        slippage = 2.0 * slippage
                    result = core.run_simulation(config.step, config.init, params, config.state_size,
                                                 scenario_bars, slippage, anchor_signal=True,
                                                 max_entry_gap=float(config.seconds))
                    trades = finalize_trades(
                        result, scenario_bars, instrument, scenario,
                        commission_pips=commission_pips(pair, price, rates, scenario.commission_base_per_lot_per_side),
                        calendar=CALENDAR,
                    )
                    if scenario is STRESS:
                        stress_sum[index[key]] += trades["r_multiple"].sum()
                        stress_count[index[key]] += len(trades)
                        continue
                    entry_seconds = scenario_bars[result[0], fx.BAR_TIME]
                    days.append(np.searchsorted(universe, fx_day_array(entry_seconds)))
                    values.append(trades["r_multiple"].to_numpy())
                    config_ids.append(np.full(len(trades), index[key]))
                    pairs_of.append(np.full(len(trades), pair, dtype=object))
                    limit = MIN_LOT_STOP_PIPS["JPY" if instrument.quote == "JPY" else "other"]
                    fits.append(np.stack([np.full(len(trades), index[key]),
                                          (trades["stop_pips"].to_numpy() <= limit)]).astype(float))
    return _assemble(keys, universe, days, values, config_ids, pairs_of, fits, stress_sum, stress_count)


def _assemble(keys, universe, days, values, config_ids, pairs_of, fits, stress_sum, stress_count) -> SampleResult:
    cat = lambda parts, empty: np.concatenate(parts) if parts else empty  # noqa: E731
    day, r = cat(days, np.empty(0, dtype=np.int64)), cat(values, np.empty(0))
    config, pair = cat(config_ids, np.empty(0, dtype=np.int64)), cat(pairs_of, np.empty(0, dtype=object))
    sums, counts = stats.daily_tables(day, r, config, len(universe), len(keys))
    pair_means = {key: {} for key in keys}
    if r.size:
        frame = pd.DataFrame({"config": config, "pair": pair, "r": r}).groupby(["config", "pair"])["r"].mean()
        for (k, name), mean in frame.items():
            pair_means[keys[k]][name] = float(mean)
    fit = np.zeros(len(keys))
    if fits:
        merged = np.concatenate(fits, axis=1)
        for k in range(len(keys)):
            selected = merged[1][merged[0] == k]
            fit[k] = selected.mean() if selected.size else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        stress_mean = stress_sum / stress_count
    return SampleResult(keys, sums, counts, pair_means, stress_mean, fit)


# --- criterion ----------------------------------------------------------------------------------


def evaluate(result: SampleResult, configs, null_t: np.ndarray, *, draws: int, min_trades: int,
             allow_inconclusive: bool) -> pd.DataFrame:
    """One row per configuration with both p-values, the numbers behind them and the verdict."""
    mean, error, t = stats.cluster_t(result.sums, result.counts)
    _, p_a = stats.bootstrap_adjusted_p_values(result.sums, result.counts, draws=draws)
    null_best = np.where(np.isfinite(null_t), null_t, -np.inf).max(axis=1)
    p_b = stats.placebo_p_value(t, null_best)
    mde = stats.minimum_detectable_effect(error)
    rows = []
    for k, key in enumerate(result.keys):
        applicable = configs[key].pairs
        means = result.pair_means[key]
        share = sum(means.get(p, -1.0) > 0 for p in applicable) / len(applicable)
        verdict = stats.judge(
            p_bootstrap=p_a[k], p_placebo=p_b[k], mean_r=mean[k], mean_r_stress=result.stress_mean[k],
            pair_share_positive=share, trades=int(result.trades[k]), mde=mde[k], min_trades=min_trades,
            allow_inconclusive=allow_inconclusive,
        )
        rows.append({"config": key, "trades": int(result.trades[k]), "mean_r": mean[k], "mean_r_stress": result.stress_mean[k],
                     "t": t[k], "p_bootstrap": p_a[k], "p_placebo": p_b[k], "pairs_positive": share, "mde": mde[k],
                     "stop_fits_min_lot": result.fits_min_lot[k], "approved": verdict.approved,
                     "inconclusive": verdict.inconclusive, "failed": "; ".join(verdict.failed)})
    return pd.DataFrame(rows)


def null_t_matrix(results: list[SampleResult]) -> np.ndarray:
    return np.stack([stats.cluster_t(r.sums, r.counts)[2] for r in results])


def false_approval_rate(results: list[SampleResult], configs, draws: int) -> float:
    """Share of placebo datasets in which the whole criterion would approve some configuration."""
    null_t = null_t_matrix(results)
    approved = 0
    for i, result in enumerate(results):
        table = evaluate(result, configs, np.delete(null_t, i, axis=0), draws=draws,
                         min_trades=stats.MIN_TRADES_DISCOVERY, allow_inconclusive=False)
        approved += bool(table["approved"].any())
    return approved / len(results)


# --- commands -------------------------------------------------------------------------------------


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607


def code_commit() -> str:
    """The commit of the code being run; refuses to run on uncommitted code."""
    if _git("status", "--porcelain", "--", "app", "scripts"):
        raise RuntimeError("uncommitted changes in app/ or scripts/: commit before running on the sample")
    return _git("rev-parse", "HEAD")


def _setup(sample: tuple[str, str], lab_dir: Path):
    manifest = pd.read_csv(DEFAULT_MANIFEST)
    matrices = open_matrices(lab_dir)
    windows = {pair: included_windows(manifest, pair, sample) for pair in (*ALL_PAIRS, SYNTHETIC)}
    windows = {pair: w for pair, w in windows.items() if w}
    return matrices, windows, day_universe(matrices, windows), median_rates(matrices)


def _placebo_worker(arguments):
    seed, sample, lab_dir = arguments
    matrices, windows, universe, rates = _setup(sample, lab_dir)
    first = int(min(m[0, fx.BAR_TIME] for m in matrices.values()) // 60)
    last = int(max(m[-1, fx.BAR_TIME] for m in matrices.values()) // 60)
    return run_sample(matrices, windows, universe, rates, configurations(),
                      coins=(placebo.coin_table(first, last, seed), first))


def run_placebos(sample, lab_dir, replicates, workers) -> list[SampleResult]:
    jobs = [(PLACEBO_SEED + i, sample, lab_dir) for i in range(replicates)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_placebo_worker, jobs))


def _plain(row: dict) -> dict:
    """A table row as JSON-safe Python values (numpy scalars converted)."""
    return {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}


def _write_report(name: str, title: str, table: pd.DataFrame, extra: str = "") -> Path:
    path = REPORT_DIR / name
    body = table.to_string(index=False, float_format=lambda value: f"{value:.4g}")
    path.write_text(f"# {title}\n\n{extra}\n\n```\n{body}\n```\n", encoding="utf-8")
    return path


def calibrate(lab_dir: Path, replicates: int, workers: int) -> int:
    commit = code_commit()
    results = run_placebos(DISCOVERY, lab_dir, replicates, workers)
    configs = configurations()
    rate = false_approval_rate(results, configs, stats.BOOTSTRAP_DRAWS)
    np.save(lab_dir / "null_t_discovery.npy", null_t_matrix(results))
    t = null_t_matrix(results)
    summary = pd.DataFrame({"config": list(configs), "median_t": np.nanmedian(t, axis=0),
                            "p95_t": np.nanpercentile(t, 95, axis=0)})
    report = _write_report("fast-calibration-report.md", "Calibração do critério em dados sem vantagem", summary,
                           f"Commit do código: {commit}. Conjuntos sintéticos: {replicates}. "
                           f"Taxa de falsa aprovação do critério inteiro: {rate:.1%} (limite {FALSE_APPROVAL_LIMIT:.0%}).")
    sys.stdout.write(f"false approval rate {rate:.3f}; report {report}\n")
    if rate > FALSE_APPROVAL_LIMIT:
        sys.stdout.write("criterion is miscalibrated: fix it before any real-data run; nothing registered\n")
        return 2
    registry.append_event({"event": "calibration_report", "false_pass_rate": rate, "replicates": replicates,
                           "code_commit": commit})
    return 0


def discover(lab_dir: Path) -> int:
    commit = code_commit()
    configs = configurations()
    matrices, windows, universe, rates = _setup(DISCOVERY, lab_dir)
    result = run_sample(matrices, windows, universe, rates, configs)
    null_t = np.load(lab_dir / "null_t_discovery.npy")
    table = evaluate(result, configs, null_t, draws=stats.BOOTSTRAP_DRAWS,
                     min_trades=stats.MIN_TRADES_DISCOVERY, allow_inconclusive=True)
    for row in table.to_dict("records"):
        registry.append_event({"event": "run", "configuration": row["config"], "sample": "discovery",
                               "code_commit": commit, "result": _plain(row)})
    approved = table.loc[table["approved"], "config"].tolist()
    report = _write_report("fast-discovery-report.md", "Descoberta (2019-01 a 2024-08)", table)
    registry.append_event({"event": "discovery_report", "approved": approved, "report": report.name})
    sys.stdout.write(f"approved: {approved}\n")
    return 0


def confirm(lab_dir: Path, replicates: int, workers: int) -> int:
    commit = code_commit()
    events = registry.read_events()
    approved = [e for e in events if e["event"] == "discovery_report"][-1]["approved"]
    if not approved:
        sys.stdout.write("nothing was approved in discovery: stop and talk to Igor\n")
        return 0
    configs = {key: config for key, config in configurations().items() if key in approved}
    null_results = run_placebos(CONFIRMATION, lab_dir, replicates, workers)
    all_keys = list(configurations())
    null_t = null_t_matrix(null_results)[:, [all_keys.index(k) for k in approved]]
    matrices, windows, universe, rates = _setup(CONFIRMATION, lab_dir)
    result = run_sample(matrices, windows, universe, rates, configs)
    table = evaluate(result, configs, null_t, draws=stats.BOOTSTRAP_DRAWS,
                     min_trades=stats.MIN_TRADES_CONFIRMATION, allow_inconclusive=False)
    for row in table.to_dict("records"):
        registry.append_event({"event": "run", "configuration": row["config"], "sample": "confirmation",
                               "code_commit": commit, "result": _plain(row)})
    _write_report("fast-confirmation-report.md", "Confirmação (2024-09 a 2026-08)", table)
    sys.stdout.write(f"confirmed: {table.loc[table['approved'], 'config'].tolist()}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["prepare", "calibrate", "discover", "confirm"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=PLACEBO_REPLICATES)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_matrices(Path("data/raw/fx_m1_hd"), LAB_DIR)
        return 0
    if args.command == "calibrate":
        return calibrate(LAB_DIR, args.replicates, args.workers)
    if args.command == "discover":
        return discover(LAB_DIR)
    return confirm(LAB_DIR, args.replicates, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
