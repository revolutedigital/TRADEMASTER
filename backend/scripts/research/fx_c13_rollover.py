"""C13-A, a flaw of the data (exploratory): the research's minutes are bid prices, and right after the 17:00 New York
rollover the spread widens and the bid sags with it. The backtest bought that sagging bid; a real buyer pays the
widened ask. This measures the sag, the banca's longs and shorts by entry time, and the money without the longs of
17:00-18:30 New York.

    python -m scripts.research.fx_c13_rollover
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_sizing as sizing
from scripts.research import fx_c14_gotobi as c14

REPORT = c3.DOCS / "c13-a-rollover.md"
ANCHOR = 16 * 60 + 50
MINUTES = (16 * 60 + 58, 17 * 60, 17 * 60 + 5, 17 * 60 + 15, 17 * 60 + 30, 18 * 60, 18 * 60 + 30, 19 * 60)
SYMBOLS = ("EURUSD", "USDJPY", "AUDNZD", "CHFJPY", "NZDCHF", "AUDCHF", "GBPCHF", "EURJPY", "GBPJPY", "AUDCAD")
PERIODS = {"2016-2022": (2016, 2022), "2023-2024": (2023, 2024), "2025-2026": (2025, 2026)}
BLOCK = (17 * 60, 18 * 60 + 30)
SCHEME = "1% em tudo (a de hoje)"


def ny_minutes(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    return np.asarray(ny.hour * 60 + ny.minute), np.asarray(ny.year)


def sag(symbol: str) -> list[dict]:
    """The bid at each minute of MINUTES against the same evening's 16:50 close, in % of the daily ATR (mean of the
    evenings): a real price has no drift of that size at a fixed minute, a widening spread does."""
    m1, fx_days, atr = c14.load_market(symbol, c3.MANIFEST, c3.ARRAYS)
    minute, year = ny_minutes(m1[:, 0])
    frame = pd.DataFrame({"evening": np.where(minute >= 17 * 60, fx_days - 1, fx_days), "minute": minute,
                          "close": m1[:, 4], "atr": atr, "year": year})
    frame = frame[frame["minute"].isin((ANCHOR, *MINUTES))]
    wide = frame.pivot_table(index="evening", columns="minute", values="close", aggfunc="last")
    evening = frame.groupby("evening")[["atr", "year"]].last()
    relative = wide.sub(wide[ANCHOR], axis=0).div(evening["atr"], axis=0) * 100
    return [{"mercado": symbol, "periodo": name, **{f"{m // 60}:{m % 60:02d}": relative.loc[keep, m].mean()
                                                  for m in MINUTES}}
            for name, (first, last) in PERIODS.items()
            for keep in [((evening["year"] >= first) & (evening["year"] <= last)).to_numpy()]]


def by_entry_time(trades: pd.DataFrame) -> pd.DataFrame:
    minute, year = ny_minutes(trades["entry_time"].to_numpy())
    table = pd.DataFrame({"entrada NY": minute // 15 * 15, "periodo": np.where(year >= 2023, "2023-26", "2016-22"),
                          "lado": np.where(trades["side"] > 0, "compra", "venda"), "net": trades["net"]})
    table = table[(table["entrada NY"] >= 16 * 60 + 45) & (table["entrada NY"] < 19 * 60 + 15)]
    out = table.groupby(["entrada NY", "periodo", "lado"])["net"].agg(["size", "mean"]).unstack(["periodo", "lado"])
    out.index = [f"{m // 60}:{m % 60:02d}" for m in out.index]
    return out


def edge(frame: pd.DataFrame, pred: np.ndarray, blocked: np.ndarray) -> pd.DataFrame:
    _, year = ny_minutes(frame["entry_time"].to_numpy())
    rows = []
    for period, (first, last) in {"2016-2022": (2016, 2022), "2023-2026": (2023, 2026)}.items():
        inside = (year >= first) & (year <= last) & np.isfinite(pred)
        days = frame.loc[inside, "fx_day"].nunique()
        for rule, keep in {"como estava": inside & (pred > 0.05),
                           "sem compras 17:00-18:30 NY": inside & (pred > 0.05) & ~blocked}.items():
            rows.append({"periodo": period, "regra": rule} | banca.score(frame[keep], days))
    return pd.DataFrame(rows)[["periodo", "regra", "trades", "mean_net", "mean_stress", "win_rate", "t"]]


def money(trades: pd.DataFrame) -> pd.DataFrame:
    minute, _ = ny_minutes(trades["entry_time"].to_numpy())
    long = (trades["side"] > 0).to_numpy()
    window = (minute >= BLOCK[0]) & (minute < BLOCK[1])
    rows = []
    for rule, drop in {"como estava": np.zeros(len(trades), bool),
                       "sem compras 17:00-17:30": long & (minute >= 17 * 60) & (minute < 17 * 60 + 30),
                       "sem compras 17:00-18:30": long & window, "sem nada 17:00-18:30": window}.items():
        kept = trades[~drop].reset_index(drop=True)
        kept.attrs = trades.attrs
        result = sizing.run(kept, schemes={SCHEME: sizing.SCHEMES[SCHEME]}, heats=(0.05,), caps=(np.inf,),
                            brakes=(0.05,), compounds=(False,))
        result["regra"] = rule
        rows.append(result)
    out = pd.concat(rows)
    out["US$/ano"] = out["ao_ano"] * 1000
    out["US$/ano estresse"] = out["ao_ano_estresse"] * 1000
    return out[["regra", "periodo", "US$/ano", "US$/ano estresse", "pior_queda", "operacoes"]]


def main() -> int:
    fmt = lambda v: f"{v:,.3g}"  # noqa: E731
    sags = pd.DataFrame([row for symbol in SYMBOLS for row in sag(symbol)])
    frame = pd.read_parquet(banca.DATASET, columns=["symbol", "side", "entry_time", "fx_day", "net", "stress"])
    reference = pd.read_parquet(banca.PREDICTIONS, columns=["row", banca.config_name(3, 2000)])
    pred = np.full(len(frame), np.nan)
    pred[reference["row"].to_numpy()] = reference[banca.config_name(3, 2000)].to_numpy()
    minute, _ = ny_minutes(frame["entry_time"].to_numpy())
    blocked = (frame["side"].to_numpy() > 0) & (minute >= BLOCK[0]) & (minute < BLOCK[1])
    accepted = pd.DataFrame({"entry_time": frame["entry_time"], "side": frame["side"], "net": frame["net"]})[
        pred > 0.05]
    text = (
        "# C13-A: o bid que afunda na virada das 17:00 de Nova York (exploratório)\n\n"
        "Os minutos da pesquisa são preço de venda (bid). Logo depois da virada o spread abre e o bid afunda junto, "
        "voltando só perto das 18:30. O backtest comprou esse bid afundado com o custo fixo de sempre; quem compra de "
        "verdade paga o ask, que subiu o mesmo tanto. Vender (ou fechar uma compra) no bid afundado é o que o mercado "
        "faz de fato, então as vendas não têm esse viés.\n\n"
        "## O bid contra o fechamento das 16:50, em % do ATR diário (média das noites)\n\n```\n"
        + sags.to_string(index=False, float_format=lambda v: f"{v:,.2f}") + "\n```\n\n"
        "## Operações aceitas pelo walk-forward, por horário de entrada (R por operação)\n\n```\n"
        + by_entry_time(accepted).to_string(float_format=fmt) + "\n```\n\n"
        "## A vantagem por operação sem as compras de 17:00-18:30\n\n```\n"
        + edge(frame, pred, blocked).to_string(index=False, float_format=lambda v: f"{v:,.4g}") + "\n```\n\n"
        "## Dinheiro com US$ 1.000 fixos (1% por operação, teto 5%, freio -5%), custo do backtest\n\n```\n"
        + money(sizing.trades()).to_string(index=False, float_format=fmt) + "\n```\n\n"
        "## Decisão\n\nO robô ao vivo não abre compra das 17:00 às 18:30 de Nova York e fecha tudo 5 minutos antes "
        "da virada (antes, 45 s): a janela e a margem saem da medição do bid, não do dinheiro. A expectativa honesta "
        "de 2023-2026 é a linha sem essas compras; a de 2016-2022 quase não muda.\n")
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
