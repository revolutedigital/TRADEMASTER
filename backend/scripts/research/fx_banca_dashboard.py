"""Local dashboard of the banca on the demo account: http://127.0.0.1:8765

    python -m scripts.research.fx_banca_dashboard

Reads the banca's journal and state, and every 20 seconds reads the demo account (balance, equity, open positions)
through its own read-only CLI session. It never sends an order.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.fx.runner.ctrader_cli import PtySession, json_of
from scripts.research import fx_banca_demo as banca
from scripts.research import fx_demo_runner as demo
from scripts.research import fx_gotobi_demo as gotobi

HOST, PORT = "127.0.0.1", 8765
REFRESH_SECONDS = 20
EQUITY_FILE = banca.DATA / "equity.jsonl"
C7_LOG, C7_REPORT = Path("/tmp/claude-1000/c7.log"), banca.c3.DOCS / "c7-report.md"
RESEARCH = [
    {"rodada": "Rodada 1", "o_que": "10 regras rápidas pré-registradas, 10 pares", "resultado": "0 de 10; todas perdem depois do custo"},
    {"rodada": "Rodada 2", "o_que": "212 combinações calibradas, 3 amostras", "resultado": "0 de 212 na descoberta"},
    {"rodada": "Codex fastN", "o_que": "~190 rodadas (FX, BTC/ETH/SOL, microestrutura)", "resultado": "nada aprovado; Fast151 aguarda dado"},
    {"rodada": "C3", "o_que": "14 estratégias publicadas, parâmetros originais", "resultado": "0 de 14 nos 10 pares"},
    {"rodada": "C4", "o_que": "TSMOM e RSI2 em 31 mercados novos", "resultado": "não aprovadas; tendência positiva em índices/metais"},
    {"rodada": "C5", "o_que": "TSMOM e RSI2 em índices (Yahoo, 2000-2026)", "resultado": "passa no período antigo, não nos índices novos"},
    {"rodada": "C6", "o_que": "banca xgboost sobre as 14, último ano", "resultado": "perda de −0,10 para −0,03 ATR; não aprovada"},
    {"rodada": "C7", "o_que": "Claude como banca, sinais diários", "resultado": "em andamento"},
    {"rodada": "C8-C9", "o_que": "banca de saída e bancas otimizadas (diário)", "resultado": "melhora real na saída; total ~0; roda na demo"},
    {"rodada": "C10", "o_que": "12 regras de day trade x 144 gestões assimétricas x 37 mercados", "resultado": "0 de 4.464"},
    {"rodada": "C11-C12", "o_que": "fluxos e calendário (fixes, noite dos EUA, virada do mês)", "resultado": "0; padrões existem antes do custo"},
    {"rodada": "C13", "o_que": "banca de day trade (xgboost) e juro/tendência em moedas", "resultado": "banca passou na descoberta e no placebo; juro 0 de 4"},
    {"rodada": "C14", "o_que": "Gotobi: compra de dólar antes do fix de Tóquio", "resultado": "APROVADA: 24 anos, confirmada em 2003-2011; roda na demo"},
]


class Broker:
    """Reads the demo account every REFRESH_SECONDS in a background thread; keeps the last snapshot."""

    def __init__(self) -> None:
        self.snapshot: dict = {"error": "ainda lendo a conta"}
        self._session: PtySession | None = None
        self._last_saved = 0.0

    def _read(self) -> dict:
        if self._session is None:
            demo.remove_container("DASH")
            self._session = PtySession(demo.cli_argv("DASH"), password=demo.password())
            self._session.start()
        positions = json_of(self._session.send("positions", timeout=60)).get("positions", [])
        for symbol in {p["symbolName"] for p in positions}:  # a price request makes the session value the position
            self._session.send(f"price {symbol}", timeout=60)
        positions = json_of(self._session.send("positions", timeout=60)).get("positions", [])
        account = json_of(self._session.send(f"account {demo.ACCOUNT}", timeout=60))
        return {"balance": account.get("balance"), "equity": account.get("equity"), "fetched_at": time.time(),
                "positions": [{k: p.get(k) for k in ("id", "symbolName", "tradeSide", "volume", "entryPrice",
                                                      "currentPrice", "netProfit", "stopLoss", "openTime")}
                              for p in positions]}

    def run(self) -> None:
        while True:
            try:
                self.snapshot = self._read()
                if time.time() - self._last_saved < 120:  # the chart keeps one point every 2 minutes
                    time.sleep(REFRESH_SECONDS)
                    continue
                self._last_saved = time.time()
                with EQUITY_FILE.open("a") as handle:
                    handle.write(json.dumps({"at": self.snapshot["fetched_at"], "equity": self.snapshot["equity"],
                                             "balance": self.snapshot["balance"]}) + "\n")
            except Exception as error:  # noqa: BLE001 - any failure: report it and log in again next time
                self.snapshot = {**self.snapshot, "error": str(error)[:200]}
                if self._session is not None:
                    self._session.close()
                    self._session = None
                    demo.remove_container("DASH")
            time.sleep(REFRESH_SECONDS)


def data(broker: Broker) -> dict:
    events = [json.loads(line) for line in banca.JOURNAL.read_text().splitlines()] if banca.JOURNAL.exists() else []
    state = json.loads(banca.STATE_FILE.read_text()) if banca.STATE_FILE.exists() else {"positions": {}}
    by_id = {str(p["id"]): p for p in broker.snapshot.get("positions", [])}
    signals = [e for e in events if e["event"] == "signal"]
    exits = [e for e in events if e["event"] == "exit"]
    per_config: dict[str, dict] = {}
    for e in signals:
        row = per_config.setdefault(e["config"], {"sinais": 0, "aceitos": 0, "fechadas": 0, "atr": 0.0, "usd": 0.0})
        row["sinais"] += 1
        row["aceitos"] += int(e["take"])
    for e in exits:
        row = per_config.setdefault(e["config"], {"sinais": 0, "aceitos": 0, "fechadas": 0, "atr": 0.0, "usd": 0.0})
        row["fechadas"] += 1
        row["atr"] += e["net"] if e["net"] == e["net"] else 0.0
        row["usd"] += e.get("profit") or 0.0
    open_rows = []
    for key, position in state["positions"].items():
        live = by_id.get(str(position["id"]), {})
        open_rows.append({"estrategia": key.split("|")[0], "mercado": position["fusion"],
                          "lado": "compra" if position["side"] == 1 else "venda", "entrada": position["entry"],
                          "atual": live.get("currentPrice"), "stop": position["stop"], "usd": live.get("netProfit"),
                          "desde": position["entry_time"]})
    equity = [json.loads(line) for line in EQUITY_FILE.read_text().splitlines()] if EQUITY_FILE.exists() else []
    progress = [line for line in C7_LOG.read_text().splitlines() if line.endswith("sinais")] if C7_LOG.exists() else []
    c7 = "relatório pronto: docs/forex/c7-report.md" if C7_REPORT.exists() else \
        f"em andamento ({progress[-1]})" if progress else "em andamento"
    gotobi_state = json.loads(gotobi.STATE.read_text()) if gotobi.STATE.exists() else {"legs": {}}
    upcoming = gotobi.upcoming_days(time.time(), gotobi.c14.japanese_holidays())[:4]
    return {
        "gotobi": {"proximos": [{"dia": str(day.date()), "compra": legs[0].open_at, "venda_final": legs[1].close_at}
                                for day in upcoming for legs in [gotobi.legs_of(day)]],
                   "pernas": [{"perna": key, "status": leg["status"], "entrada": leg.get("entry"),
                               "usd": leg.get("profit"), "motivo": leg.get("reason", "")}
                              for key, leg in sorted(gotobi_state["legs"].items(), reverse=True)]},
        "now": time.time(), "broker": {k: v for k, v in broker.snapshot.items() if k != "positions"},
        "dia": state.get("guard", {}) | {"meta": banca.DAILY_TARGET_USD, "limite": banca.DAILY_STOP_USD},
        "totais": {"sinais": len(signals), "aceitos": sum(int(e["take"]) for e in signals), "fechadas": len(exits),
                   "abertas": len(open_rows), "usd_fechado": sum(e.get("profit") or 0.0 for e in exits),
                   "usd_aberto": sum(r["usd"] or 0.0 for r in open_rows)},
        "abertas": sorted(open_rows, key=lambda r: -r["desde"]), "por_estrategia": per_config,
        "ultimos_sinais": [{"quando": e["at"], "estrategia": e["config"], "mercado": banca.FUSION.get(e["symbol"], e["symbol"]),
                            "lado": "compra" if e["side"] == 1 else "venda", "previsao": e["prediction"],
                            "decisao": "entrou" if e["take"] else "pulou"} for e in signals[-40:]][::-1],
        "fechadas": [{"quando": e["exit_time"], "estrategia": e["config"], "mercado": banca.FUSION.get(e["symbol"], e["symbol"]),
                      "atr": e["net"], "usd": e.get("profit"), "motivo": e["reason"]} for e in exits][::-1],
        "equity": [{"t": e["at"], "v": e["equity"]} for e in equity if e.get("equity") is not None],
        "pesquisa": [r | ({"resultado": c7} if r["rodada"] == "C7" else {}) for r in RESEARCH],
    }


PAGE = (Path(__file__).with_name("fx_banca_dashboard.html")).read_text(encoding="utf-8")


def serve() -> None:
    broker = Broker()
    threading.Thread(target=broker.run, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path.startswith("/data.json"):
                body, kind = json.dumps(data(broker), default=float).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    serve()
