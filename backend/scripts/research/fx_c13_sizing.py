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

import itertools
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
def ledger(entry_t, exit_t, net, per_unit, min_vol, step, risk, day, asset_a, sign_a, asset_b, sign_b, n_assets,
           heat, currency_cap, brake, start, tolerance=1.0, compound=False):
    """Every share (the trade's risk, `heat`, `currency_cap`) is of one base: the realized balance at entry when
    `compound` (what the live runner does with its equity), else the starting balance for good (profits taken out,
    losses topped up). Taken at the minimum volume when that risks at most `tolerance` times the trade's budget, else
    skipped; skipped when the open stops would pass `heat`, when one asset's open stops in one direction would pass
    `currency_cap`, or once the FX day's realized result has reached -`brake` of the day's first balance. Entries sorted by time (ties in the order they come:
    the one first gets the room); `asset_b` < 0 for an index. Returns each trade's result in US$ (0 if refused), then
    what `simulate` returns."""
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
        base = balance if compound else start
        volume = np.floor(risk[i] * base / per_unit[i] / step[i] + 1e-9) * step[i]
        if volume < min_vol[i] - 1e-12:
            if min_vol[i] * per_unit[i] > tolerance * risk[i] * base:
                n_small += 1
                continue
            volume = min_vol[i]
        stop_usd = volume * per_unit[i]
        if open_risk + stop_usd > heat * base:
            n_heat += 1
            continue
        a, b = asset_a[i], asset_b[i]
        if abs(exposure[a] + sign_a[i] * stop_usd) > currency_cap * base or \
                (b >= 0 and abs(exposure[b] + sign_b[i] * stop_usd) > currency_cap * base):
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
    return pnl, balance, worst, worst_open, n_taken, n_small, n_heat, n_currency, n_brake


@njit(cache=True)
def simulate(entry_t, exit_t, net, per_unit, min_vol, step, risk, day, asset_a, sign_a, asset_b, sign_b, n_assets,
             heat, currency_cap, brake, start, tolerance=1.0, compound=False):
    """`ledger` without the per-trade results: final balance, worst drawdown of the realized balance, worst drawdown
    had every open stop been hit at once, trades taken, skipped for the minimum volume, the heat cap, the currency
    cap, the daily brake."""
    result = ledger(entry_t, exit_t, net, per_unit, min_vol, step, risk, day, asset_a, sign_a, asset_b, sign_b,
                    n_assets, heat, currency_cap, brake, start, tolerance, compound)
    return result[1:]


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


def run(frame: pd.DataFrame, schemes: dict = SCHEMES, heats=HEATS, caps=CURRENCY_CAPS, brakes=DAILY_BRAKES,
        tolerances=(1.0,), priorities=(False,), compounds=(True, False)) -> pd.DataFrame:
    """Every combination of the rules, in every period, from US$ 1,000. With `prioridade`, trades entering at the same
    minute are offered the room in order of the model's prediction, highest first (else in the order the data holds
    them). `compostos`: every share of the balance at entry (the live runner), or of the US$ 1,000 for good.
    `ao_ano`: the yearly return (compound rate when compounding, else simple); `primeiro_ano`: the first 12 months."""
    rows = []
    for priority in priorities:
        ordered = frame.sort_values(["entry_time", "pred"], ascending=[True, False], kind="stable") \
            if priority else frame
        side = ordered["side"].to_numpy(dtype=np.float64)
        arrays = {
            "per_unit": ordered["per_unit"].to_numpy(np.float64), "min_vol": ordered["min_vol"].to_numpy(np.float64),
            "step": ordered["step"].to_numpy(np.float64), "day": ordered["fx_day"].to_numpy(np.int64),
            "asset_a": ordered["asset_a"].to_numpy(np.int64), "asset_b": ordered["asset_b"].to_numpy(np.int64),
            "sign_a": side, "sign_b": -side}
        for period in sweep.PERIODS:
            start, end, years = sweep.period_bounds(period)
            inside = ((ordered["entry_time"] >= start) & (ordered["entry_time"] < end)).to_numpy()
            part = {k: v[inside] for k, v in arrays.items()}
            times = [ordered.loc[inside, c].to_numpy(np.float64) for c in ("entry_time", "exit_time")]
            pred = ordered.loc[inside, "pred"].to_numpy()
            for name, scheme in schemes.items():
                risk = risk_of(pred, scheme)
                keep = risk > 0
                sub = {k: v[keep] for k, v in part.items()}
                t = [x[keep] for x in times]
                first = t[0] < start + 365 * 86400
                results = {c: ordered.loc[inside, c].to_numpy(np.float64)[keep] for c in ("net", "stress")}
                for heat, cap, brake, tolerance, compound in itertools.product(heats, caps, brakes, tolerances,
                                                                                compounds):
                    def go(cost: str, rows_: np.ndarray | slice = slice(None)):
                        return simulate(t[0][rows_], t[1][rows_], results[cost][rows_], sub["per_unit"][rows_],
                                        sub["min_vol"][rows_], sub["step"][rows_], risk[keep][rows_],
                                        sub["day"][rows_], sub["asset_a"][rows_], sub["sign_a"][rows_],
                                        sub["asset_b"][rows_], sub["sign_b"][rows_], frame.attrs["n_assets"], heat,
                                        cap, brake, sweep.START_USD, tolerance, compound)
                    final, worst, worst_open, taken, small, capped, currency, braked = go("net")
                    stress = go("stress")[0]
                    rate = (lambda v: (v / sweep.START_USD) ** (1 / years) - 1 if v > 0 else -1.0) if compound else \
                        (lambda v: (v - sweep.START_USD) / sweep.START_USD / years)
                    rows.append({
                        "periodo": period, "esquema": name, "teto_aberto": heat, "teto_moeda": cap, "freio_dia": brake,
                        "prioridade": priority, "tolerancia": tolerance, "compostos": compound,
                        "ao_ano": rate(final), "ao_ano_estresse": rate(stress),
                        "primeiro_ano": go("net", first)[0] / sweep.START_USD - 1,
                        "primeiro_ano_estresse": go("stress", first)[0] / sweep.START_USD - 1,
                        "pior_queda": worst, "pior_queda_todos_stops": worst_open, "operacoes": taken,
                        "puladas_volume": small, "puladas_teto": capped, "puladas_moeda": currency,
                        "puladas_freio": braked})
    return pd.DataFrame(rows)


RULES = ["esquema", "teto_aberto", "teto_moeda", "freio_dia"]
RULES_2 = [*RULES, "prioridade", "tolerancia"]
LIVE = ("1% / 2% / 3% pela confiança", 0.075, 0.03, 0.05, True, 1.0)  # the rules running since 2026-09-27
REFINE = {"esquema": ["1% / 2% / 3% pela confiança", "1% / 1,5% / 2% pela confiança", "1% em tudo (a de hoje)"],
          "teto_aberto": list(HEATS), "teto_moeda": list(CURRENCY_CAPS), "freio_dia": [np.inf, 0.05],
          "prioridade": [False, True], "tolerancia": [1.0, 1.5]}
REPORT_2, RESULTS_2 = c3.DOCS / "c13-a-sizing-2.md", banca.LAB / "sizing2_money.parquet"
SHOWN = ["ao_ano", "ao_ano_estresse", "primeiro_ano", "primeiro_ano_estresse", "pior_queda", "pior_queda_todos_stops"]
CORRECTION = (
    "**Corrigido em 2026-09-27.** A primeira versão dimensionava cada operação sobre os US$ 1.000 iniciais mas media "
    "os tetos (stops abertos, por moeda) sobre o saldo, que crescia: com a conta maior cabiam cada vez mais operações, "
    "e o lucro por ano saiu muito inflado (US$ 2.000-2.800 por ano sobre US$ 1.000, quando o primeiro ano rendia ~20%). "
    "Agora cada simulação usa uma base só: o saldo do momento (`compostos` = True, o que o robô faz) ou os US$ 1.000 "
    "para sempre (`compostos` = False, lucro retirado e perda reposta).\n\n")


def _matches(money: pd.DataFrame, setting: dict) -> np.ndarray:
    mask = np.ones(len(money), dtype=bool)
    for k, v in setting.items():
        column = money[k].to_numpy()
        mask &= np.isinf(column.astype(float)) if isinstance(v, float) and np.isinf(v) else column == v
    return mask


def honest(money: pd.DataFrame, rules: list[str], reference: tuple, size: int = 12) -> pd.DataFrame:
    """The `size` rules with the best 2016-2022 yearly return at double cost within DD_CAP of worst drawdown (every
    open stop hit at once), beside their own 2023-2026, then the reference rules."""
    early = money[(money["periodo"] == "2016-2022") & (money["pior_queda_todos_stops"] <= DD_CAP)]
    late = money[money["periodo"] == "2023-2026"].set_index(rules)
    picks = list(early.sort_values("ao_ano_estresse", ascending=False).head(size).iterrows())
    ref = money[(money["periodo"] == "2016-2022") & _matches(money, dict(zip(rules, reference)))]
    rows = []
    for label, (_, best) in [(f"{i + 1}ª", p) for i, p in enumerate(picks)] + [("referência", (0, ref.iloc[0]))]:
        after = late.loc[tuple(best[k] for k in rules)]
        rows.append({"posição": label, **{k: best[k] for k in rules},
                     "16-22_ano": best["ao_ano"], "16-22_estresse": best["ao_ano_estresse"],
                     "16-22_queda": best["pior_queda_todos_stops"],
                     "23-26_ano": after["ao_ano"], "23-26_estresse": after["ao_ano_estresse"],
                     "23-26_1º_ano": after["primeiro_ano"], "23-26_queda": after["pior_queda_todos_stops"]})
    return pd.DataFrame(rows)


def one_at_a_time(money: pd.DataFrame, rules: list[str], reference: tuple, grid: dict) -> pd.DataFrame:
    """Each rule changed alone from the reference ones, in 2016-2022 and 2023-2026."""
    base = dict(zip(rules, reference))
    rows = []
    for rule, values in grid.items():
        for value in values:
            for _, row in money[_matches(money, base | {rule: value})].iterrows():
                if row["periodo"] != "2016-2026":
                    rows.append({"regra": rule, "valor": value, "periodo": row["periodo"],
                                 **{k: row[k] for k in SHOWN}, "operacoes": row["operacoes"]})
    return pd.DataFrame(rows).sort_values(["regra", "periodo"], kind="stable")


def _table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n"


def _sections(money: pd.DataFrame, rules: list[str], reference: tuple, grid: dict) -> str:
    text = ""
    for compound, label in ((True, "reinvestindo (tudo sobre o saldo do momento — o que o robô faz)"),
                            (False, "sem reinvestir (tudo sobre os US$ 1.000 para sempre)")):
        part = money[money["compostos"] == compound]
        text += (f"## {label}\n\n### Uma regra de cada vez, a partir da referência\n\n"
                 f"{_table(one_at_a_time(part, rules, reference, grid))}\n"
                 f"### As 12 melhores de 2016-2022 (custo dobrado, queda ≤ {DD_CAP:.0%}), cada uma em 2023-2026\n\n"
                 f"{_table(honest(part, rules, reference))}\n")
    return text


INTRO = ("Entradas e saídas congeladas (as escolhas do walk-forward, stop 1 ATR, alvo 3R, fim do dia FX); só muda "
         "quanto cada operação arrisca e quando uma operação é recusada. Conta de US$ 1.000, volume mínimo real de "
         "cada mercado na demo. `ao_ano`: retorno por ano (taxa composta quando reinveste, simples quando não); "
         "`primeiro_ano`: o que os US$ 1.000 rendem nos 12 primeiros meses do período; `pior_queda`: do saldo "
         "realizado; `pior_queda_todos_stops`: se todo stop aberto batesse junto. `teto_moeda`: a soma máxima de stops "
         "abertos numa mesma moeda e direção (inf = sem teto). `freio_dia`: sem novas entradas no resto do dia FX "
         "depois de perder essa fração no dia (inf = sem freio).\n\n"
         "**Isto não aprova nada.** As regras são escolhidas só com 2016-2022 e lidas em 2023-2026.\n\n")


def write_report(money: pd.DataFrame) -> str:
    grid = {"esquema": list(SCHEMES), "teto_aberto": HEATS, "teto_moeda": CURRENCY_CAPS, "freio_dia": DAILY_BRAKES}
    text = ("# C13, frente A: regras de dinheiro (exploratório, não é teste pré-registrado)\n\n" + CORRECTION + INTRO
            + f"Referência: {CURRENT[0]}, stops abertos até {CURRENT[1]:.0%}, sem teto por moeda, sem freio.\n\n"
            + _sections(money, RULES, CURRENT, grid))
    REPORT.write_text(text, encoding="utf-8")
    return text


def write_report_2(money: pd.DataFrame) -> str:
    text = ("# C13, frente A: regras de dinheiro, rodada 2 (exploratório, não é teste pré-registrado)\n\n" + CORRECTION
            + "Duas regras novas: `prioridade` — operações que entram no mesmo minuto recebem a vaga dos tetos em ordem "
            "da previsão do modelo, maior primeiro; `tolerancia` — quando o volume mínimo arrisca mais que o orçamento, "
            "entra com o mínimo se ele arriscar até essa vez o orçamento (1 = pula).\n\n" + INTRO
            + "Referência: as regras que rodam na demo (1% / 2% / 3% pela confiança, stops abertos até 7,5%, 3% por "
            "moeda, freio a -5% do dia, prioridade).\n\n" + _sections(money, RULES_2, LIVE, REFINE))
    REPORT_2.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["run"])[0]
    frame = trades()
    sys.stdout.write(f"{len(frame):,} trades, {frame.attrs['n_assets']} assets\n")
    if command == "refine":
        schemes = {name: SCHEMES[name] for name in REFINE["esquema"]}
        money = run(frame, schemes, REFINE["teto_aberto"], REFINE["teto_moeda"], REFINE["freio_dia"],
                    REFINE["tolerancia"], REFINE["prioridade"])
        money.to_parquet(RESULTS_2)
        sys.stdout.write(write_report_2(money))
        return 0
    money = run(frame)
    money.to_parquet(RESULTS)
    sys.stdout.write(write_report(money))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
