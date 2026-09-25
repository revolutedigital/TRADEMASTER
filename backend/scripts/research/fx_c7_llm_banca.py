"""Round C7: Claude (the subscription, headless `claude -p`) as the banca on the daily signals of C6's test year.

See docs/forex/c7-preregistration.md. Context is anonymised: no market name, date or price reaches the model.

    python -m scripts.research.fx_c7_llm_banca run

Nothing here touches the trading engine, the database, or an exchange.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6

DOCS = c3.DOCS
DAILY = ("TURTLE20", "TURTLE55", "SQUEEZE_D1", "NR7_D1", "CHANDELIER_D1")
BATCH, SEED, TRIES, MIN_ACCEPTED = 25, 20260926, 3, 50
CACHE = c3.LAB / "c7_answers.jsonl"
CLASS_NAMES = {"major": "câmbio (par principal)", "cross": "câmbio (cruzado)", "metal": "metal", "index": "índice de ações",
               "energy": "energia", "dollar_index": "índice do dólar"}
DESCRIPTIONS = {
    "TURTLE20": "rompimento da máxima/mínima de 20 dias (Turtle 1), stop 2 ATR, sai no canal de 10 dias",
    "TURTLE55": "rompimento de 55 dias (Turtle 2), stop 2 ATR, sai no canal de 20 dias",
    "SQUEEZE_D1": "rompimento das bandas de Bollinger depois de compressão de volatilidade, stop na banda oposta",
    "NR7_D1": "rompimento da barra de menor range dos últimos 7 dias, stop no outro extremo, sai em 3 dias",
    "CHANDELIER_D1": "cruzamento das médias 50/200 com stop móvel de 3 ATR",
}
INSTRUCTIONS = """Você é a mesa de risco de um fundo quantitativo. Cada linha abaixo é um sinal de uma estratégia
técnica publicada, num mercado anônimo, no fechamento de um dia. O resultado de cada operação é medido em ATR e já
paga o custo de operar. Na média histórica essas estratégias PERDEM depois do custo; seu trabalho é aceitar só os
sinais em que o contexto torna o resultado esperado positivo, e pular o resto. Aceitar pouco é aceitável.

Campos: estrategia, lado (+1 comprado, -1 vendido), classe do mercado, custo_atr (custo da operação em ATR),
vol_d1 (ATR14/ATR100 diário: >1 volatilidade acima do normal), vol_h1 (o mesmo em barras de 1h), trend50 e trend200
(distância do fechamento às médias de 50 e 200 dias, em ATR), slope50 (inclinação da média de 50 em 5 dias, em ATR),
*_side (o mesmo multiplicado pelo lado: positivo = operação a favor da tendência), rsi14 (diário), ret4h_side (retorno
das últimas 4h a favor do lado, em ATR de 1h), hora (Nova York), dia_semana (0 = segunda), recent_same (média das
últimas 10 operações dessa estratégia neste mercado, em ATR), recent_config (média das últimas 50 dessa estratégia em
todos os mercados).

Estratégias:
{descriptions}

Responda SOMENTE com JSON, sem texto em volta: [{{"id": <id>, "decisao": "entrar" ou "pular"}}, ...] com uma entrada
para cada id.

Sinais:
{rows}"""


def anonymised(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for index, row in frame.iterrows():
        item = {"id": int(index), "estrategia": row["config"], "lado": int(row["side"]),
                "classe": CLASS_NAMES[c6.instrument_class(row["symbol"])]}
        for key in ("cost_atr", "vol_d1", "vol_h1", "trend50", "trend200", "slope50", "trend50_side", "trend200_side",
                    "slope50_side", "rsi14", "ret4h_side", "recent_same", "recent_config"):
            value = row[key]
            item[key] = None if pd.isna(value) else round(float(value), 3)
        item["hora"], item["dia_semana"] = int(row["hour"]), int(row["weekday"])
        rows.append(item)
    return rows


def parse(text: str, ids: set[int]) -> dict[int, bool] | None:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        answers = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    decisions = {int(a["id"]): a.get("decisao") == "entrar" for a in answers if isinstance(a, dict) and "id" in a}
    return decisions if set(decisions) >= ids else None


def ask(rows: list[dict]) -> dict[int, bool]:
    ids = {r["id"] for r in rows}
    prompt = INSTRUCTIONS.format(descriptions="\n".join(f"- {k}: {v}" for k, v in DESCRIPTIONS.items()),
                                 rows="\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    for _ in range(TRIES):
        try:
            result = subprocess.run(["claude", "-p", prompt, "--model", "opus"], capture_output=True, text=True,  # noqa: S603, S607
                                    timeout=900, check=False)
        except subprocess.TimeoutExpired:
            continue  # a slow answer counts as a failed try
        decisions = parse(result.stdout, ids)
        if decisions is not None:
            return {i: decisions[i] for i in ids}
    return dict.fromkeys(ids, False)


def run() -> int:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603, S607
    if subprocess.run(["git", "status", "--porcelain", "--", "scripts", "app"], capture_output=True, text=True, check=True).stdout:  # noqa: S603, S607
        raise RuntimeError("commit the code before running")
    from xgboost import XGBRegressor

    table = c6.build_trades(pd.read_csv(c3.MANIFEST))
    train = table[table["exit_time"] < c6.TEST_START]
    test = table[(table["entry_time"] >= c6.TEST_START) & (table["entry_time"] < c6.TEST_END)].copy()
    model = XGBRegressor(**c6.MODEL_PARAMS)
    model.fit(train[c6.FEATURES].to_numpy(dtype=float), train["net"].to_numpy())
    test["xgb"] = model.predict(test[c6.FEATURES].to_numpy(dtype=float)) > 0
    signals = test[(test["group"] == "B") & test["config"].isin(DAILY)]
    order = np.random.default_rng(SEED).permutation(signals.index.to_numpy())
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            done.update({int(k): v for k, v in json.loads(line).items()})
    for start in range(0, len(order), BATCH):
        chunk = [i for i in order[start:start + BATCH] if int(i) not in done]
        if not chunk:
            continue
        answers = ask(anonymised(signals.loc[chunk]))
        done.update(answers)
        with CACHE.open("a") as handle:
            handle.write(json.dumps({str(k): v for k, v in answers.items()}) + "\n")
        sys.stdout.write(f"{min(start + BATCH, len(order))}/{len(order)} sinais\n")
        sys.stdout.flush()
    signals = signals.assign(claude=[done[int(i)] for i in signals.index])
    accepted = signals[signals["claude"]]
    s_acc = c6.summary(accepted)
    approved = bool(s_acc["trades"] >= MIN_ACCEPTED and s_acc["mean_atr"] > 0 and s_acc["mean_atr_stress"] > 0
                    and s_acc["p_one_sided"] <= 0.05)
    rows = [{"conjunto": "todos os sinais diários"} | c6.summary(signals),
            {"conjunto": "aceitos pelo Claude"} | s_acc,
            {"conjunto": "aceitos pela banca do C6 (xgboost)"} | c6.summary(signals[signals["xgb"]]),
            {"conjunto": "aceitos pelos dois"} | c6.summary(signals[signals["claude"] & signals["xgb"]])]
    fmt = lambda v: f"{v:.4g}"  # noqa: E731
    by_config = signals.groupby("config").agg(sinais=("net", "size"), aceitos=("claude", "sum"))
    report = DOCS / "c7-report.md"
    report.write_text(
        f"# C7: o Claude como banca (sinais diários, grupo B, último ano)\n\nCommit do código: {commit}. "
        f"**Aprovado: {approved}.** Resultado em ATR por operação, líquido do custo.\n\n"
        f"```\n{pd.DataFrame(rows).to_string(index=False, float_format=fmt)}\n```\n\n```\n{by_config.to_string()}\n```\n",
        encoding="utf-8")
    doc = DOCS / "c7-preregistration.md"
    with (DOCS / "c7-registry.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"date": date.today().isoformat(), "event": "c7_report", "code_commit": commit,
                                 "preregistration_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                                 "approved": approved, "accepted": s_acc}, sort_keys=True) + "\n")
    sys.stdout.write(report.read_text())
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["run"]:
        sys.stdout.write("usage: python -m scripts.research.fx_c7_llm_banca run\n")
        raise SystemExit(2)
    raise SystemExit(run())
