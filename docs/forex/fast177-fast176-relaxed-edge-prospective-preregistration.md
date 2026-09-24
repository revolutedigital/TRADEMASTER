# Fast177 — Fast176 relaxed-edge prospective shadow preregistration

Fast177 is a new prospective offline shadow derived from the Fast176 diagnostic
observation stream.

Fast176 remains frozen and unchanged. Fast177 does not amend Fast176, does not
approve a strategy, and does not authorize Demo, Testnet or LIVE execution.

## Preregistration anchor

Only decisions strictly after this timestamp can count:

`2026-09-24T10:05:09.742000-03:00`

Equivalent UTC timestamp:

`2026-09-24T13:05:09.742000+00:00`

Any data before this anchor is diagnostic only.

## Why this exists

The first Fast176 prospective decisions showed two useful facts:

- the strict Fast176 gate avoided multiple poor LONG decisions;
- the strict Fast176 gate also missed profitable LONG moves because the edge and
  regime thresholds were too conservative for the observed fresh market window.

Fast177 freezes one relaxed shadow to test that observation honestly in future
data only.

## Frozen policy

Target:

- `ETHUSDT`.

Model:

- `Ridge(alpha=10.0)`;
- `SimpleImputer(strategy="constant", fill_value=0.0)`;
- `StandardScaler`;
- train only on `fast50`, `fast51`, `fast52`.

Input data:

- Binance USD-M public `aggTrades`;
- symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`;
- one-minute causal features;
- decision stride: `5m`;
- horizon: `900s`.

Entry rule:

```text
abs(prediction) >= 7.5
ETHUSDT_qv_30m >= 100000000.0
```

Direction:

- `LONG` if `prediction > 0`;
- `SHORT` if `prediction < 0`.

Cost:

- `2.0` bps per offline decision.

## Gates

Early failure:

- at least `20` eligible decisions and total net bps `<= -100.0`; or
- any sufficient day with at least `5` eligible decisions and day net bps
  `<= -60.0`.

Shadow candidate gate:

- at least `60` eligible decisions;
- at least `5` days;
- net mean bps `> 3.0`;
- net total bps `> 0`;
- positive sufficient-day share `>= 0.60`;
- one-sided lower confidence bound `> 0`;
- no sufficient-day loss breach.

Even if the shadow candidate gate passes, Fast177 remains offline research until
explicit human review and separate authorization.

## Safety

Fast177 must always report:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

No order submission, no ledger commit, no Demo/Testnet/LIVE activation.
