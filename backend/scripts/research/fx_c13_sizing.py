"""C13 front A, exploratory study of the money rules (NOT a pre-registered test, nothing here approves anything): the
entries and exits stay the frozen ones (the walk-forward's picks, stop 1 ATR, target 3R, end of the FX day); only how
much each trade risks, and when a trade is refused, changes. From a US$ 1,000 account, no compounding, with the demo
broker's real minimum volumes (data/c13/specs.json) and the trades of fx_c13_sweep (run it first).

The rules:
- risk by confidence: each trade's stop is worth a share of the account that grows with the model's prediction;
- a cap on the stops open at once (the live runner's 5%);
- a cap per currency: the stops open on one currency in one direction (long EURUSD is long EUR and short USD; an
  index is its own asset) may not pass a share of the balance — a burst of picks on one dollar move is one bet;
- a daily brake: no new entry for the rest of the FX day once the day's realized result reaches -X% (open positions
  are not closed: the backtest has no price path to close them on, and closing trades with a positive expectancy
  early is itself a cost).

The honest check: the rules are chosen on 2016-2022 only and read on 2023-2026, which the choice never saw.

    python -m scripts.research.fx_c13_sizing
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from numba import njit

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_demo as live
from scripts.research import fx_c13_sweep as sweep

REPORT = c3.DOCS / "c13-a-sizing.md"
RESULTS = banca.LAB / "sizing_money.parquet"
# prediction floors of the buckets and the risk of each: the first floor is the entry threshold
SCHEMES = {
    "1% em tudo (a de hoje)": ((0.05, 0.01),),
    "1,5% em tudo": ((0.05, 0.015),),
    "2% em tudo": ((0.05, 0.02),),
    "1% / 1,5% / 2% pela confiança": ((0.05, 0.01), (0.08, 0.015), (0.12, 0.02)),
    "1% / 2% / 3% pela confiança": ((0.05, 0.01), (0.08, 0.02), (0.12, 0.03)),
    "0,5% abaixo de 0,05 + 1% acima": ((0.02, 0.005), (0.05, 0.01)),
    "0,5% / 1% / 1,5% / 2% pela confiança": ((0.02, 0.005), (0.05, 0.01), (0.08, 0.015), (0.12, 0.02)),
}
HEATS = (0.05, 0.075, 0.10)
CURRENCY_CAPS = (np.inf, 0.03, 0.02)
DAILY_BRAKES = (np.inf, 0.05, 0.03, 0.02)
CURRENT = ("1% em tudo (a de hoje)", 0.05, np.inf, np.inf)
DD_CAP = 0.25


@njit(cache=True)
def simulate(entry_t, exit_t, net, per_unit, min_vol, step, risk, day, asset_a, sign_a, asset_b, sign_b, n_assets,
             heat, currency_cap, brake, start):
    """Sized on the starting balance (no compounding); skipped when the minimum volume risks more than the trade's
    budget, when the open stops would pass `heat` of the balance, when one asset's open stops in one direction would
    pass `currency_cap` of it, or once the FX day's realized result has reached -`brake` of the day's first balance.
    Entries sorted by time; `asset_b` < 0 for an index. Returns final balance, worst drawdown of the realized
    balance, worst drawdown had every open stop been hit at once, trades taken, skipped for the minimum volume, for
    the heat cap, for the currency cap, for the daily brake."""
    n = entry_t.shape[0]
    order = np.argsort(exit_t)
    pnl = np.zeros(n)
    at_risk = np.zeros(n)
    taken = np.zeros(n, dtype=np.bool_)
    exposure = np.zeros(n_assets)
    balance, peak, worst, worst_open, open_risk = start, start, 0.0, 0.0, 0.0
    n_taken, n_small, n_heat, n_currency, n_brake, j = 0, 0, 0, 0, 0, 0
    current_day, day_start = -1, start
    for i in range(n + 1):
        now = entry_t[i] if i < n else np.inf
        while j < n and exit_t[order[j]] < now:
            k = order[j]
            if taken[k]:
                balance += pnl[k]
                open_risk -= at_risk[k]
                exposure[asset_a[k]] -= sign_a[k] * at_risk[k]
                if asset_b[k] >= 0:
                    exposure[asset_b[k]] -= sign_b[k] * at_risk[k]
                peak = max(peak, balance)
                worst = max(worst, (peak - balance) / peak)
            j += 1
        if i == n or balance <= 0:
            break
        if day[i] != current_day:
            current_day, day_start = day[i], balance
        if balance - day_start <= -brake * day_start:
            n_brake += 1
            continue
        volume = np.floor(risk[i] * start / per_unit[i] / step[i] + 1e-9) * step[i]
        if volume < min_vol[i] - 1e-12:
            n_small += 1
            continue
        stop_usd = volume * per_unit[i]
        if open_risk + stop_usd > heat * balance:
            n_heat += 1
            continue
        a, b = asset_a[i], asset_b[i]
        if abs(exposure[a] + sign_a[i] * stop_usd) > currency_cap * balance or \
                (b >= 0 and abs(exposure[b] + sign_b[i] * stop_usd) > currency_cap * balance):
            n_currency += 1
            continue
        taken[i] = True
        n_taken += 1
        at_risk[i] = stop_usd
        pnl[i] = net[i] * stop_usd
        open_risk += stop_usd
        exposure[a] += sign_a[i] * stop_usd
        if b >= 0:
            exposure[b] += sign_b[i] * stop_usd
        worst_open = max(worst_open, (peak - (balance - open_risk)) / peak)
    return balance, worst, worst_open, n_taken, n_small, n_heat, n_currency, n_brake


def trades() -> pd.DataFrame:
    """The frozen management's trades above 0.02 (their busy chain is the 0.02 threshold's; the rules then pick
    among them), with what the simulation needs."""
    table = pd.read_parquet(sweep.CANDIDATES_FILE)
    rows = pd.read_parquet(sweep.TRADES)
    rows = rows[(rows["g"] == sweep.FROZEN_G) & (rows["tau_i"] == sweep.THRESHOLDS.index(0.02))]
    sides = pd.read_parquet(banca.DATASET, columns=["symbol", "set_id", "entry_time", "side"])
    table = table.merge(sides, on=["symbol", "set_id", "entry_time"], how="left")
    specs = json.loads(sweep.SPECS.read_text())
    table = table.assign(usd=sweep.usd_per_quote(table), min_vol=table["symbol"].map(lambda s: specs[s]["minVolume"]),
                         step=table["symbol"].map(lambda s: specs[s]["volumeStep"]))
    frame = rows.merge(table, on="cid", suffixes=("", "_candidate"))
    frame = frame[frame["net"].abs() <= sweep.SUSPECT_R].sort_values("entry_time").reset_index(drop=True)
    frame["per_unit"] = frame["atr"] * frame["usd"]  # the frozen stop is 1 daily ATR
    assets, codes = [], {}
    for symbol in frame["symbol"]:
        assets.append(tuple(codes.setdefault(p, len(codes)) if p else -1 for p in live.assets_of(symbol)))
    frame["asset_a"], frame["asset_b"] = zip(*assets)
    frame.attrs["n_assets"] = len(codes)
    return frame


def risk_of(pred: np.ndarray, scheme: tuple) -> np.ndarray:
    risk = np.zeros(len(pred))
    for floor, share in scheme:
        risk[pred > floor] = share
    return risk


def run(frame: pd.DataFrame) -> pd.DataFrame:
    side = frame["side"].to_numpy(dtype=np.float64)
    arrays = {
        "per_unit": frame["per_unit"].to_numpy(np.float64), "min_vol": frame["min_vol"].to_numpy(np.float64),
        "step": frame["step"].to_numpy(np.float64), "day": frame["fx_day"].to_numpy(np.int64),
        "asset_a": frame["asset_a"].to_numpy(np.int64), "asset_b": frame["asset_b"].to_numpy(np.int64),
        "sign_a": side, "sign_b": -side}
    rows = []
    for period in sweep.PERIODS:
        start, end, years = sweep.period_bounds(period)
        inside = ((frame["entry_time"] >= start) & (frame["entry_time"] < end)).to_numpy()
        part = {k: v[inside] for k, v in arrays.items()}
        times = [frame.loc[inside, c].to_numpy(np.float64) for c in ("entry_time", "exit_time")]
        pred = frame.loc[inside, "pred"].to_numpy()
        for name, scheme in SCHEMES.items():
            risk = risk_of(pred, scheme)
            keep = risk > 0
            sub = {k: v[keep] for k, v in part.items()}
            t = [x[keep] for x in times]
            results = {c: frame.loc[inside, c].to_numpy(np.float64)[keep] for c in ("net", "stress")}
            for heat in HEATS:
                for cap in CURRENCY_CAPS:
                    for brake in DAILY_BRAKES:
                        args = (sub["per_unit"], sub["min_vol"], sub["step"], risk[keep], sub["day"], sub["asset_a"],
                                sub["sign_a"], sub["asset_b"], sub["sign_b"], frame.attrs["n_assets"], heat, cap,
                                brake, sweep.START_USD)
                        final, worst, worst_open, taken, small, capped, currency, braked = simulate(
                            *t, results["net"], *args)
                        stress = simulate(*t, results["stress"], *args)[0]
                        rows.append({"periodo": period, "esquema": name, "teto_aberto": heat, "teto_moeda": cap,
                                     "freio_dia": brake, "lucro_ano_usd": (final - sweep.START_USD) / years,
                                     "lucro_ano_estresse_usd": (stress - sweep.START_USD) / years,
                                     "pior_queda": worst, "pior_queda_todos_stops": worst_open, "operacoes": taken,
                                     "puladas_volume": small, "puladas_teto": capped, "puladas_moeda": currency,
                                     "puladas_freio": braked})
    return pd.DataFrame(rows)


RULES = ["esquema", "teto_aberto", "teto_moeda", "freio_dia"]


def honest(money: pd.DataFrame, size: int = 12) -> pd.DataFrame:
    """The `size` rules that made the most in 2016-2022 at double cost within DD_CAP of worst drawdown, beside
    their own 2023-2026, then the live rules for reference."""
    early = money[(money["periodo"] == "2016-2022") & (money["pior_queda_todos_stops"] <= DD_CAP)]
    late = money[money["periodo"] == "2023-2026"].set_index(RULES)
    picks = list(early.sort_values("lucro_ano_estresse_usd", ascending=False).head(size).iterrows())
    current = money[(money["periodo"] == "2016-2022") & (money["esquema"] == CURRENT[0]) &
                    (money["teto_aberto"] == CURRENT[1]) & np.isinf(money["teto_moeda"]) & np.isinf(money["freio_dia"])]
    rows = []
    for label, (_, best) in [(f"{i + 1}ª", p) for i, p in enumerate(picks)] + [("a de hoje", (0, current.iloc[0]))]:
        after = late.loc[tuple(best[k] for k in RULES)]
        rows.append({"posição": label, **{k: best[k] for k in RULES},
                     "16-22_ano": best["lucro_ano_usd"], "16-22_estresse": best["lucro_ano_estresse_usd"],
                     "16-22_queda": best["pior_queda_todos_stops"],
                     "23-26_ano": after["lucro_ano_usd"], "23-26_estresse": after["lucro_ano_estresse_usd"],
                     "23-26_queda": after["pior_queda_todos_stops"]})
    return pd.DataFrame(rows)


def one_at_a_time(money: pd.DataFrame) -> pd.DataFrame:
    """Each rule changed alone from the live ones, both periods."""
    base = dict(zip(RULES, CURRENT))
    rows = []
    for rule, values in (("esquema", list(SCHEMES)), ("teto_aberto", HEATS), ("teto_moeda", CURRENCY_CAPS),
                         ("freio_dia", DAILY_BRAKES)):
        for value in values:
            setting = base | {rule: value}
            mask = np.ones(len(money), dtype=bool)
            for k, v in setting.items():
                mask &= (money[k] == v).to_numpy() if not (isinstance(v, float) and np.isinf(v)) else \
                    np.isinf(money[k].to_numpy())
            for _, row in money[mask].iterrows():
                if row["periodo"] == "2016-2026":
                    continue
                rows.append({"regra": rule, "valor": value, "periodo": row["periodo"],
                             "lucro_ano_usd": row["lucro_ano_usd"], "estresse": row["lucro_ano_estresse_usd"],
                             "queda_todos_stops": row["pior_queda_todos_stops"], "operacoes": row["operacoes"]})
    return pd.DataFrame(rows).sort_values(["regra", "periodo"], kind="stable")


def write_report(money: pd.DataFrame) -> str:
    table = lambda frame: "```\n" + frame.to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n"  # noqa: E731
    text = (
        "# C13, frente A: regras de dinheiro (exploratório, não é teste pré-registrado)\n\n"
        "Entradas e saídas congeladas (as escolhas do walk-forward, stop 1 ATR, alvo 3R, fim do dia FX); só muda "
        "quanto cada operação arrisca e quando uma operação é recusada. Conta de US$ 1.000, sem juros compostos, "
        "volume mínimo real de cada mercado na demo; lucro por ano em US$ sobre os US$ 1.000; `queda_todos_stops`: a "
        "pior queda se todo stop aberto batesse junto. `teto_moeda`: a soma máxima de stops abertos numa mesma moeda "
        "e direção (inf = sem teto). `freio_dia`: sem novas entradas no resto do dia FX depois de perder essa fração "
        "no dia (inf = sem freio; as posições abertas não são fechadas).\n\n"
        "**Isto não aprova nada.** As regras são escolhidas só com 2016-2022 e lidas em 2023-2026.\n\n"
        f"## 1. Uma regra de cada vez, a partir da de hoje ({CURRENT[0]}, teto aberto {CURRENT[1]:.0%})\n\n"
        f"{table(one_at_a_time(money))}\n"
        f"## 2. As 12 melhores de 2016-2022 (custo dobrado, queda ≤ {DD_CAP:.0%}), cada uma em 2023-2026\n\n"
        f"{table(honest(money))}\n")
    REPORT.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    frame = trades()
    sys.stdout.write(f"{len(frame):,} trades, {frame.attrs['n_assets']} assets\n")
    money = run(frame)
    money.to_parquet(RESULTS)
    sys.stdout.write(write_report(money))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
