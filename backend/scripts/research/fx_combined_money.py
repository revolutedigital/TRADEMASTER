"""C13-A and Gotobi on one US$ 1,000 account (exploratory, not a pre-registered test): how much of the account each
Gotobi leg should risk next to the day-trade banca. C13-A runs its live money rules (docs/forex/c13-a-sizing-2.md);
each Gotobi leg (configuration B of C14, on USDJPY) risks a share of the starting US$ 1,000 per daily ATR, in whole
minimum volumes of 1,000 units and never less than one (what the live runner does). No compounding; results are
booked when each trade ends; the drawdown is that of the account's realized balance.

The Gotobi risk is chosen on 2016-2022 only and read on 2023-2026. Run fx_c13_sweep first.

    python -m scripts.research.fx_combined_money
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from scripts.research import fx_banca_demo as old_banca
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_demo as live
from scripts.research import fx_c13_sizing as sizing
from scripts.research import fx_c13_sweep as sweep
from scripts.research import fx_c14_gotobi as c14

REPORT = c3.DOCS / "c13-c14-combined.md"
GOTOBI_RISKS = {"desligado": 0.0, "lote mínimo (hoje)": -1.0, "1%": 0.01, "2%": 0.02, "3%": 0.03, "4%": 0.04,
                "5%": 0.05}
DD_CAP = 0.25


def c13_results() -> pd.DataFrame:
    """C13-A's trades under the live rules (priority by prediction): exit time and US$ result, per period."""
    frame = sizing.trades()
    ordered = frame.sort_values(["entry_time", "pred"], ascending=[True, False], kind="stable")
    rows = []
    for period in sweep.PERIODS:
        start, end, _ = sweep.period_bounds(period)
        part = ordered[(ordered["entry_time"] >= start) & (ordered["entry_time"] < end)]
        risk = np.array([live.risk_for(p) for p in part["pred"]])
        part, risk = part[risk > 0], risk[risk > 0]
        side = part["side"].to_numpy(np.float64)
        common = (part["per_unit"].to_numpy(np.float64), part["min_vol"].to_numpy(np.float64),
                  part["step"].to_numpy(np.float64), risk, part["fx_day"].to_numpy(np.int64),
                  part["asset_a"].to_numpy(np.int64), side, part["asset_b"].to_numpy(np.int64), -side,
                  frame.attrs["n_assets"], live.MAX_OPEN_RISK, live.MAX_ASSET_RISK, live.DAILY_BRAKE, sweep.START_USD)
        times = [part[c].to_numpy(np.float64) for c in ("entry_time", "exit_time")]
        for cost in ("net", "stress"):
            pnl = sizing.ledger(*times, part[cost].to_numpy(np.float64), *common)[0]
            taken = pnl != 0
            rows.append(pd.DataFrame({"periodo": period, "custo": cost, "time": times[1][taken], "usd": pnl[taken],
                                      "estrategia": "C13"}))
    return pd.concat(rows, ignore_index=True)


def gotobi_legs() -> pd.DataFrame:
    """Every Gotobi leg of 2012-2026: when it ends, its R (normal and double cost) and the US$ one daily ATR is worth
    per unit of USDJPY at its entry."""
    holidays = c14.japanese_holidays()
    markets, rates = c14.sample(c3.MANIFEST, c3.ARRAYS, *c14.DISCOVERY, c14.carry.download_rates(cached_only=True))
    m1, fx_days, atr = markets["USDJPY"]
    rows = []
    for day in c14.gotobi_days(*c14.DISCOVERY, holidays):
        midnight = (day - pd.Timedelta(seconds=c14.JST)).timestamp()
        for _, side, start, exit_bar, _ in c14.LEGS["B"]:
            result = c14.leg_result(m1, fx_days, atr, rates, "USDJPY", side, midnight + start, midnight + exit_bar,
                                    c3.cost_of("USDJPY"))
            if result is None:
                continue
            e = c14.entry_minute(m1[:, 0], midnight + start)
            rows.append({"entry_time": m1[e, 0], "time": midnight + exit_bar + 60, "net": result["net"],
                         "stress": result["stress"], "per_unit": atr[e] / m1[e, 1]})
    return pd.DataFrame(rows)


def gotobi_results(legs: pd.DataFrame, risk: float) -> pd.DataFrame:
    minimum = float(json.loads(sweep.SPECS.read_text())["USDJPY"]["minVolume"])
    if risk == 0:
        return pd.DataFrame(columns=["periodo", "custo", "time", "usd", "estrategia"])
    rows = []
    for period in sweep.PERIODS:
        start, end, _ = sweep.period_bounds(period)
        part = legs[(legs["entry_time"] >= start) & (legs["entry_time"] < end)]
        units = np.array([minimum if risk < 0 else
                          (old_banca.position_volume(risk * sweep.START_USD, u, minimum, minimum) or minimum)
                          for u in part["per_unit"]])
        per_r = units * part["per_unit"].to_numpy()
        for cost in ("net", "stress"):
            rows.append(pd.DataFrame({"periodo": period, "custo": cost, "time": part["time"].to_numpy(),
                                      "usd": part[cost].to_numpy() * per_r, "estrategia": "Gotobi"}))
    return pd.concat(rows, ignore_index=True)


def account(results: pd.DataFrame, years: float) -> dict:
    """Profit per year and the worst drawdown of the realized balance of a US$ 1,000 account."""
    ordered = results.sort_values("time")
    balance = sweep.START_USD + ordered["usd"].cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate(([sweep.START_USD], balance)))[1:]
    worst = float(((peak - balance) / peak).max()) if len(balance) else 0.0
    return {"lucro_ano_usd": float(ordered["usd"].sum()) / years, "pior_queda": worst}


def study() -> pd.DataFrame:
    c13 = c13_results()
    legs = gotobi_legs()
    rows = []
    for name, risk in GOTOBI_RISKS.items():
        gotobi = gotobi_results(legs, risk)
        for period in sweep.PERIODS:
            _, _, years = sweep.period_bounds(period)
            row = {"gotobi": name, "periodo": period}
            for cost, label in (("net", ""), ("stress", "_estresse")):
                both = pd.concat([c13[(c13["periodo"] == period) & (c13["custo"] == cost)],
                                  gotobi[(gotobi["periodo"] == period) & (gotobi["custo"] == cost)]])
                combined = account(both, years)
                row |= {f"lucro_ano{label}": combined["lucro_ano_usd"], f"queda{label}": combined["pior_queda"]}
                if cost == "net" and risk != 0:
                    alone = account(gotobi[(gotobi["periodo"] == period) & (gotobi["custo"] == cost)], years)
                    row |= {"gotobi_sozinho_ano": alone["lucro_ano_usd"], "gotobi_sozinho_queda": alone["pior_queda"]}
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(table: pd.DataFrame) -> str:
    early = table[(table["periodo"] == "2016-2022") & (table["queda"] <= DD_CAP)]
    best = early.loc[early["lucro_ano_estresse"].idxmax(), "gotobi"]
    fmt = lambda frame: "```\n" + frame.to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n"  # noqa: E731
    text = (
        "# C13-A e Gotobi na mesma conta (exploratório, não é teste pré-registrado)\n\n"
        "Conta de US$ 1.000, sem juros compostos. C13-A com as regras que rodam na demo (1% / 2% / 3% pela "
        "confiança, stops abertos até 7,5%, 3% por moeda, freio a -5% do dia, prioridade pela confiança). Cada perna "
        "do Gotobi (configuração B, USDJPY) arrisca a fração indicada dos US$ 1.000 por ATR diário, em lotes "
        "inteiros de 1.000 unidades e nunca menos que um (o que o robô faz); `lote mínimo (hoje)` é o que roda "
        "hoje. Lucro por ano em US$; `queda`: a pior queda do saldo realizado da conta, com as duas juntas.\n\n"
        f"Escolhido só com 2016-2022 (maior lucro com custo dobrado e queda ≤ {DD_CAP:.0%}): **{best}**.\n\n"
        f"{fmt(table.sort_values(['periodo', 'gotobi'], kind='stable'))}\n")
    REPORT.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    table = study()
    table.to_parquet(banca.LAB / "combined_money.parquet")
    sys.stdout.write(write_report(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
