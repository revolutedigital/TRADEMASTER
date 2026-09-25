"""The banca on the Fusion DEMO account, forward only: the 14 strategies of C3 raise their hand on live bars and the
frozen C6 model decides, every hour, which signals to take. Orders are the minimum volume of each symbol.

    python -m scripts.research.fx_banca_demo train      # freeze the model on everything up to 2026-08 (once)
    python -m scripts.research.fx_banca_demo run        # run until ~/.config/trademaster/STOP exists
    python -m scripts.research.fx_banca_demo report     # what it has done so far
    touch ~/.config/trademaster/RESTART_BANCA              # exit keeping the positions (restart to load new code)

Deviation from the backtest, declared: strategies without a stop in the original (TSMOM, RSI2) get a disaster stop
of 5 ATR on the server, because nothing runs on the account without a server-side stop.

Only the demo account (10139135) is used: the session is pinned to it by `fx_demo_runner.cli_argv`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.fx.runner.ctrader_cli import CliVenue, PtySession, json_of
from app.fx.runner.venue import OrderRejected, VenueUnavailable
from scripts.research import fx_c3_lab as c3
from scripts.research import fx_c6_lab as c6
from scripts.research import fx_demo_runner as demo

DATA = Path("data/banca_demo")
MODEL_FILE, SEED_FILE = DATA / "model.json", DATA / "seed.parquet"
STATE_FILE, JOURNAL = DATA / "state.json", DATA / "journal.jsonl"
STOP_FILE = demo.STOP_FILE
RESTART_FILE = STOP_FILE.with_name("RESTART_BANCA")  # exit without closing anything, to load new code
TRAIN_END = pd.Timestamp("2026-09-01", tz="UTC").timestamp()
FUSION = {s: s for s in c3.GROUP_A + c3.GROUP_B if len(s) == 6 and s not in c6.CLASSES} | {
    "XAUUSD": "XAUUSD", "XAGUSD": "XAGUSD", "SPXUSD": "US500", "NSXUSD": "NAS100", "GRXEUR": "GER40",
    "UKXGBP": "UK100", "JPXJPY": "JPN225", "FRXEUR": "FRA40", "AUXAUD": "AUS200", "HKXHKD": "HK50",
    "WTIUSD": "XTIUSD", "BCOUSD": "XBRUSD", "UDXUSD": "USDX"}
EXCLUDED = frozenset({"XAUUSD", "XAGUSD", "WTIUSD", "BCOUSD"})  # Igor, 2026-09-25: their minimum volume is too big for
# a US$ 1,000 account (US$ 40 to 130 per daily ATR against a US$ 10 daily stop)
PERIOD = {"H1": 3600, "H4": 14400, "D1": 86400}
CLI_PERIOD = {"H1": "h1", "H4": "h4", "D1": "d1"}
BARS = 400
DISASTER_STOP_ATR = 5.0
CHECK_SECOND = 30  # process the bars that closed on the hour, at HH:00:30
SYNC_SECONDS = 60
RISK_PER_ATR = 0.005  # every position is sized so that one daily ATR move is worth 0.5% of the equity
TAKE_MINIMUM_ABOVE_BUDGET = True  # Igor, 2026-09-25: all 41 markets, even where the minimum volume is worth more
DAILY_TARGET_USD, DAILY_STOP_USD = 20.0, 10.0  # Igor, 2026-09-25: close everything and pause until the next FX day
GUARD_SECONDS = 20
LOCK_TRAIL_ATR = 1.0  # Igor, 2026-09-25: at the target, bank the biggest winners and trail the rest 1 daily ATR behind
QUOTE = {"US500": "USD", "NAS100": "USD", "USDX": "USD", "GER40": "EUR", "FRA40": "EUR", "UK100": "GBP",
         "JPN225": "JPY", "AUS200": "AUD", "HK50": "HKD", "XAUUSD": "USD", "XAGUSD": "USD", "XTIUSD": "USD",
         "XBRUSD": "USD"}
USD_PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF", "USDHKD")


# --- training -----------------------------------------------------------------------------------------------------


def train() -> int:
    from xgboost import XGBRegressor

    DATA.mkdir(parents=True, exist_ok=True)
    table = c6.build_trades(pd.read_csv(c3.MANIFEST))
    fit = table[table["exit_time"] < TRAIN_END]
    model = XGBRegressor(**c6.MODEL_PARAMS)
    model.fit(fit[c6.FEATURES].to_numpy(dtype=float), fit["net"].to_numpy())
    model.save_model(str(MODEL_FILE))
    seed = fit.sort_values("exit_time").groupby(["config", "symbol"]).tail(50)
    seed[["config", "symbol", "entry_time", "exit_time", "net"]].to_parquet(SEED_FILE)
    digest = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()
    sys.stdout.write(f"model trained on {len(fit):,} trades; sha256 {digest}\n")
    return 0


# --- pure helpers (tested) ------------------------------------------------------------------------------------------


def closed_bars(bars: list[dict], period: int, now: float) -> np.ndarray:
    """(open time, open, high, low, close) of the bars that have closed by `now`."""
    rows = []
    for bar in bars:
        opened = datetime.fromisoformat(bar["timestamp"].replace("Z", "+00:00")).timestamp()
        if opened + period <= now:
            rows.append((opened, bar["open"], bar["high"], bar["low"], bar["close"]))
    return np.array(rows, dtype=float).reshape(-1, 5)


def protective_levels(side: int, price: float, plan: c3.Plan, j: int) -> tuple[float, float | None]:
    """Stop and target for an entry at `price`, as the backtest sets them (disaster stop when the rule has none)."""
    if not np.isnan(plan.stop_level[j]):
        stop = float(plan.stop_level[j])
    elif not np.isnan(plan.stop_distance[j]):
        stop = price - side * float(plan.stop_distance[j])
    else:
        stop = price - side * DISASTER_STOP_ATR * float(plan.unit[j])
    target = None if np.isnan(plan.target_r) else price + side * plan.target_r * abs(price - stop)
    return stop, target


def position_volume(budget_usd: float, usd_per_unit_per_atr: float, minimum: float, step: float) -> float | None:
    """The largest volume (a multiple of the step) whose daily ATR move is worth at most the budget; None if even
    the minimum volume is worth more (the position is not taken rather than taken bigger than the others)."""
    steps = np.floor(budget_usd / usd_per_unit_per_atr / step + 1e-9)
    volume = round(float(steps * step), 8)
    if volume < minimum:
        return None
    return int(volume) if float(volume).is_integer() else volume


def winners_to_bank(profits: list[tuple[str, float]], target: float) -> list[str]:
    """Biggest winners first, until their profits add up to the target (losers are never banked)."""
    chosen, banked = [], 0.0
    for key, profit in sorted(profits, key=lambda item: -item[1]):
        if banked >= target or profit <= 0:
            break
        chosen.append(key)
        banked += profit
    return chosen


def tightened_stop(side: int, best: float, atr: float, stop: float, price: float, trail_atr: float) -> float | None:
    """The new stop `trail_atr` ATR behind the best price, or None if it would not tighten or is not behind the price."""
    candidate = best - side * trail_atr * atr
    tighter = candidate > stop if side == 1 else candidate < stop
    behind = candidate < price if side == 1 else candidate > price
    return candidate if tighter and behind else None


def usd_per_currency(mids: dict[str, float]) -> dict[str, float]:
    return {"USD": 1.0, "EUR": mids["EURUSD"], "GBP": mids["GBPUSD"], "AUD": mids["AUDUSD"], "NZD": mids["NZDUSD"],
            "JPY": 1 / mids["USDJPY"], "CAD": 1 / mids["USDCAD"], "CHF": 1 / mids["USDCHF"], "HKD": 1 / mids["USDHKD"]}


def recent_features(history: pd.DataFrame, config: str, symbol: str, now: float) -> tuple[float, float]:
    same = history[(history["config"] == config) & (history["symbol"] == symbol)]
    every = history[history["config"] == config]
    value = lambda frame, window: c6.recent_mean(np.array([now]), frame["exit_time"].to_numpy(float),  # noqa: E731
                                                 frame["net"].to_numpy(float), window)[0] if len(frame) else np.nan
    return value(same, 10), value(every, 50)


# --- runner ---------------------------------------------------------------------------------------------------------


class Banca:
    def __init__(self, session: PtySession) -> None:
        from xgboost import XGBRegressor

        self.session, self.venue = session, CliVenue(session, demo.ACCOUNT)
        self.model = XGBRegressor()
        self.model.load_model(str(MODEL_FILE))
        self.model_sha = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()
        self.state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"positions": {}, "processed": {}}
        self.specs: dict[str, dict] = {}

    def log(self, event: str, **fields) -> None:
        with JOURNAL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": time.time(), "event": event, **fields}, default=float, sort_keys=True) + "\n")

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, indent=1))

    def history(self) -> pd.DataFrame:
        seed = pd.read_parquet(SEED_FILE)
        exits = [e for e in self._events() if e["event"] == "exit"]
        live = pd.DataFrame([{k: e[k] for k in ("config", "symbol", "entry_time", "exit_time", "net")} for e in exits])
        return pd.concat([seed, live], ignore_index=True) if len(live) else seed

    @staticmethod
    def _events() -> list[dict]:
        return [json.loads(line) for line in JOURNAL.read_text().splitlines()] if JOURNAL.exists() else []

    def send(self, command: str) -> dict:
        return json_of(self.session.send(command, timeout=60))

    def spec(self, fusion: str) -> dict:
        if fusion not in self.specs:
            self.specs[fusion] = self.send(f"symbol {fusion}")
        return self.specs[fusion]

    def sized_volume(self, fusion: str, d1: np.ndarray) -> tuple[float | None, float]:
        """Volume for one daily ATR = RISK_PER_ATR of the equity, and the dollars one ATR is worth per unit."""
        atr = float(c3.atr(d1[:, 1], d1[:, 2], d1[:, 3], d1[:, 4], 20)[-1])
        per_unit = atr * self.usd_per[QUOTE.get(fusion, fusion[3:])]
        spec = self.spec(fusion)
        volume = position_volume(RISK_PER_ATR * self.equity, per_unit, float(spec["minVolume"]), float(spec["volumeStep"]))
        if volume is None and TAKE_MINIMUM_ABOVE_BUDGET:
            minimum = float(spec["minVolume"])
            volume = int(minimum) if minimum.is_integer() else minimum
        return volume, per_unit

    async def process(self, now: float) -> None:
        history = self.history()
        mids = {}
        for pair in USD_PAIRS:
            quote = await self.venue.quote(pair)
            mids[pair] = 0.5 * (quote.bid + quote.ask)
        self.usd_per = usd_per_currency(mids)
        self.equity = (await self.venue.account()).equity
        for symbol, fusion in FUSION.items():
            if symbol in EXCLUDED:
                continue
            bars = {tf: closed_bars(self.send(f"candles {fusion} {CLI_PERIOD[tf]} {BARS}")["bars"], PERIOD[tf], now)
                    for tf in PERIOD}
            if min(len(b) for b in bars.values()) < 260:
                continue
            for config, (tf, build) in c3.STRATEGIES.items():
                try:
                    await self._decide(config, tf, build, symbol, fusion, bars, history, now)
                except OrderRejected as error:  # one refused order must not stop the other decisions of the hour
                    self.log("error", config=config, symbol=symbol, message=str(error)[:300])
        self.save()

    async def _decide(self, config, tf, build, symbol, fusion, bars, history, now) -> None:
        b = bars[tf]
        marker = f"{symbol}|{tf}|{config}"
        if self.state["processed"].get(marker) == b[-1, 0]:
            return
        self.state["processed"][marker] = b[-1, 0]
        o, h, l, c = b[:, 1], b[:, 2], b[:, 3], b[:, 4]
        plan = build(o, h, l, c)
        j = len(b) - 1
        key = f"{config}|{symbol}"
        position = self.state["positions"].get(key)
        if position is not None:
            position = await self._manage(key, position, plan, j, h[j], l[j], fusion)
        if position is None and plan.enter[j] != 0 and not np.isnan(plan.unit[j]):
            await self._consider(config, symbol, fusion, int(plan.enter[j]), plan, j, bars, history, now)

    async def _manage(self, key, position, plan, j, high, low, fusion):
        side = position["side"]
        position["held"] += 1
        if not np.isnan(plan.trail_k) and not np.isnan(plan.trail_atr[j]):
            position["best"] = max(position["best"], high) if side == 1 else min(position["best"], low)
            candidate = position["best"] - side * plan.trail_k * plan.trail_atr[j]
            if (side == 1 and candidate > position["stop"]) or (side == -1 and candidate < position["stop"]):
                candidate = round(candidate, int(self.spec(fusion)["digits"]))
                await self.venue.amend_protection(position["id"], stop_price=candidate, target_price=position["target"])
                position["stop"] = candidate
                self.log("trail", key=key, stop=candidate)
        rule_exit = (side == 1 and (plan.exit_long[j] or plan.enter[j] == -1)) or \
                    (side == -1 and (plan.exit_short[j] or plan.enter[j] == 1)) or \
                    (plan.max_bars > 0 and position["held"] >= plan.max_bars)
        if rule_exit:
            await self.venue.close(position["id"])
            await self._record_exit(key, position, "rule")
            return None
        return position

    async def _consider(self, config, symbol, fusion, side, plan, j, bars, history, now) -> None:
        d1, h1 = bars["D1"], bars["H1"]
        features = c6.features_from_bars(d1[:, 1:], h1[:, 1:], np.array([len(d1) - 1]), np.array([len(h1) - 1]),
                                         np.array([now]), np.array([float(side)]))
        recent_same, recent_config = recent_features(history, config, symbol, now)
        row = features.iloc[0].to_dict() | {
            "config_id": list(c3.STRATEGIES).index(config), "side": float(side),
            "class_id": ["major", "cross", "metal", "index", "energy", "dollar_index"].index(c6.instrument_class(symbol)),
            "cost_atr": c3.cost_of(symbol) / plan.unit[j], "recent_same": recent_same, "recent_config": recent_config}
        prediction = float(self.model.predict(np.array([[row[f] for f in c6.FEATURES]], dtype=float))[0])
        take = prediction > 0
        self.log("signal", config=config, symbol=symbol, side=side, prediction=prediction, take=take,
                 model_sha=self.model_sha)
        if not take:
            return
        volume, per_unit = self.sized_volume(fusion, d1)
        if volume is None:
            self.log("refused", config=config, symbol=symbol,
                     reason=f"o volume mínimo vale US$ {per_unit * float(self.spec(fusion)['minVolume']):.2f} por ATR, "
                            f"acima de {RISK_PER_ATR:.1%} do patrimônio")
            return
        quote = await self.venue.quote(fusion)
        price = quote.ask if side == 1 else quote.bid
        digits = int(self.spec(fusion)["digits"])
        stop, target = protective_levels(side, price, plan, j)
        stop, target = round(stop, digits), None if target is None else round(target, digits)
        if (side == 1 and stop >= price) or (side == -1 and stop <= price):
            self.log("refused", config=config, symbol=symbol, reason="stop behind the price")
            return
        key = f"{config}|{symbol}"
        position = await self.venue.market_order(symbol=fusion, side=side, units=volume, stop_price=stop,
                                                 target_price=target, client_order_id=f"{key}|{int(now)}")
        if position.stop_price is None:
            await self.venue.close(position.id)
            self.log("refused", config=config, symbol=symbol, reason="no server-side stop")
            return
        self.state["positions"][key] = {"id": position.id, "fusion": fusion, "side": side, "entry": position.entry_price,
                                        "entry_time": now, "unit": float(plan.unit[j]), "stop": stop, "target": target,
                                        "held": 0, "best": position.entry_price,
                                        "atr_d1": float(c3.atr(d1[:, 1], d1[:, 2], d1[:, 3], d1[:, 4], 20)[-1])}
        self.log("entry", key=key, position_id=position.id, entry=position.entry_price, stop=stop, target=target,
                 volume=volume, usd_per_atr=volume * per_unit)
        self.save()

    async def _record_exit(self, key, position, reason) -> None:
        closed = await self.venue.exit_of(position["id"], position["fusion"])
        exit_price = closed.price if closed else float("nan")
        config, symbol = key.split("|")
        net = (position["side"] * (exit_price - position["entry"]) - c3.cost_of(symbol)) / position["unit"]
        self.log("exit", key=key, config=config, symbol=symbol, position_id=position["id"], reason=reason,
                 entry_time=position["entry_time"], exit_time=closed.time if closed else time.time(), net=net,
                 exit=exit_price, profit=closed.result if closed else None)
        self.state["positions"].pop(key, None)
        self.save()

    async def sync(self) -> None:
        """Positions the server closed on its own (stop, target) are journaled."""
        open_ids = {p.id for p in await self.venue.positions()}
        for key, position in list(self.state["positions"].items()):
            if position["id"] not in open_ids:
                await self._record_exit(key, position, "server")

    def paused(self) -> bool:
        return self.state.get("guard", {}).get("status", "operando") != "operando"

    async def guard(self, now: float) -> None:
        """The day's target and stop on the whole account: reached, everything closes until the next FX day."""
        day = int(c3.lab.fx_day_array(np.array([now]))[0])
        equity = (await self.venue.account()).equity
        guard = self.state.setdefault("guard", {})
        if guard.get("day") != day:
            guard.update(day=day, start_equity=equity, status="operando")
            self.state["processed"] = {}  # a new day re-evaluates every strategy's last bar
            self.log("new_day", day=day, start_equity=equity)
            self.save()
            return
        guard["equity"] = equity
        result = equity - guard["start_equity"]
        if guard["status"] == "operando" and result <= -DAILY_STOP_USD:
            guard["status"] = "limite do dia"
            await self.flatten(guard["status"])
            guard["closed_result"] = (await self.venue.account()).equity - guard["start_equity"]
            self.log("day_closed", status=guard["status"], result=result, closed_result=guard["closed_result"])
        elif guard["status"] == "operando" and result >= DAILY_TARGET_USD:
            await self.bank_winners(guard, result)
        elif guard["status"] == "meta: segurando o resto":
            await self.trail_rest(guard)
        self.save()

    async def live_positions(self) -> dict[str, dict]:
        """Open positions as the broker values them now (a price request makes the session mark them)."""
        raw = json_of(self.session.send("positions", timeout=60)).get("positions", [])
        for symbol in {p["symbolName"] for p in raw}:
            self.session.send(f"price {symbol}", timeout=60)
        raw = json_of(self.session.send("positions", timeout=60)).get("positions", [])
        return {str(p["id"]): p for p in raw}

    async def bank_winners(self, guard: dict, result: float) -> None:
        """Target reached: close the biggest winners until the target is banked; the rest trail 1 daily ATR behind."""
        live = await self.live_positions()
        mine = [(key, pos, live[pos["id"]]) for key, pos in self.state["positions"].items() if pos["id"] in live]
        profits = {key: raw.get("netProfit") or 0.0 for key, _, raw in mine}
        banked = 0.0
        for key in winners_to_bank(list(profits.items()), DAILY_TARGET_USD):
            position = self.state["positions"][key]
            await self.venue.close(position["id"])
            await self._record_exit(key, position, "meta: maior ganho realizado")
            banked += profits[key]
        guard["status"] = "meta: segurando o resto" if self.state["positions"] else "meta do dia"
        guard["banked"] = banked
        self.log("target", result=result, banked=banked, kept=len(self.state["positions"]))
        await self.trail_rest(guard)

    async def trail_rest(self, guard: dict) -> None:
        """Each kept position's stop follows its best price since the target, 1 daily ATR behind; it only tightens."""
        if not self.state["positions"]:
            guard["status"] = "meta do dia"
            return
        live = await self.live_positions()
        for key, position in list(self.state["positions"].items()):
            raw = live.get(position["id"])
            if raw is None or raw.get("currentPrice") is None:
                continue
            side, price = position["side"], float(raw["currentPrice"])
            atr = position.get("atr_d1") or position["unit"]
            position["trail_best"] = max(position.get("trail_best", price), price) if side == 1 else \
                min(position.get("trail_best", price), price)
            candidate = tightened_stop(side, position["trail_best"], atr, position["stop"], price, LOCK_TRAIL_ATR)
            if candidate is not None:
                candidate = round(candidate, int(self.spec(position["fusion"])["digits"]))
                await self.venue.amend_protection(position["id"], stop_price=candidate, target_price=position["target"])
                position["stop"] = candidate
                self.log("lock_trail", key=key, stop=candidate)

    async def flatten(self, reason: str) -> None:
        for key, position in list(self.state["positions"].items()):
            await self.venue.close(position["id"])
            await self._record_exit(key, position, reason)


async def loop() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    demo.remove_container("BANCA")
    session = PtySession(demo.cli_argv("BANCA"), password=demo.password())
    try:
        await asyncio.to_thread(session.start)
        banca = Banca(session)
        await banca.sync()
        banca.log("start", model_sha=banca.model_sha)
        last_hour, last_sync, last_guard = None, 0.0, 0.0
        while True:
            now = time.time()
            if RESTART_FILE.exists():  # leave the positions and their server-side stops as they are
                RESTART_FILE.unlink()
                banca.log("restart")
                return 0
            if STOP_FILE.exists():
                await banca.flatten("stop file")
                banca.log("stop")
                return 0
            if now - last_guard >= GUARD_SECONDS:
                try:
                    await banca.guard(now)
                except (VenueUnavailable, OrderRejected) as error:
                    banca.log("error", message=str(error)[:300])
                last_guard = now
            hour = int(now // 3600)
            if not banca.state["processed"]:
                last_hour = None  # a new day (or a fresh start) evaluates at once
            if hour != last_hour and (now % 3600 >= CHECK_SECOND or last_hour is None) and not banca.paused():
                try:
                    await banca.process(now)
                except (VenueUnavailable, OrderRejected) as error:
                    banca.log("error", message=str(error)[:300])
                last_hour = hour
            if now - last_sync >= SYNC_SECONDS:
                await banca.sync()
                last_sync = now
            await asyncio.sleep(5)
    finally:
        session.close()
        demo.remove_container("BANCA")


def report() -> int:
    events = [json.loads(line) for line in JOURNAL.read_text().splitlines()] if JOURNAL.exists() else []
    signals = [e for e in events if e["event"] == "signal"]
    exits = [e for e in events if e["event"] == "exit"]
    taken = sum(e["take"] for e in signals)
    net = np.array([e["net"] for e in exits if np.isfinite(e["net"])])
    money = sum(e["profit"] for e in exits if e.get("profit") is not None)
    sys.stdout.write(f"sinais: {len(signals)}, aceitos pela banca: {taken}, operações fechadas: {len(exits)}, "
                     f"média {net.mean() if net.size else float('nan'):+.3f} ATR, resultado na demo US$ {money:+.2f}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    command = (argv or sys.argv[1:] or ["run"])[0]
    if command == "train":
        return train()
    if command == "report":
        return report()
    return asyncio.run(loop())


if __name__ == "__main__":
    raise SystemExit(main())
