# Fast178 — Fast177 edge-2 prospective shadow preregistration

Fast178 is a new prospective offline shadow derived from the Fast177 observation
stream. It does not amend Fast177 and does not authorize Demo, Testnet or LIVE
execution.

## Preregistration anchor

Only decisions strictly after this timestamp can count:

`2026-09-24T10:42:20.437000-03:00`

Equivalent UTC timestamp:

`2026-09-24T13:42:20.437000+00:00`

Any data before this anchor is diagnostic only.

## Why this exists

Fast177 relaxed the Fast176 gate to `abs(prediction) >= 7.5` and
`ETHUSDT_qv_30m >= 100000000.0`.

The first Fast177 decisions showed that the relaxed regime gate was active, but
the edge cutoff still missed positive movement and the first eligible trade did
not cover explicit cost. This suggests the model's raw prediction may be more
useful as a direction/regime signal than as a monotonic payoff ranking at the
`7.5` bps cutoff.

Fast178 freezes one lower edge floor for future-only observation:

- keep the same model and regime gate;
- lower the entry edge floor to the explicit-cost scale: `2.0` bps.

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
abs(prediction) >= 2.0
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

Even if the shadow candidate gate passes, Fast178 remains offline research until
explicit human review and separate authorization.

## Safety

Fast178 must always report:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

No order submission, no ledger commit, no Demo/Testnet/LIVE activation.
