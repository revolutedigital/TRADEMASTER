# Fast183 — Fast177 edge-7.5 with micro_8_3 management preregistration

Fast183 is a future-only offline observation derived from Fast182's best
historical survivor. It does not amend Fast177/Fast182 and does not authorize
Demo, Testnet or LIVE execution.

## Preregistration anchor

Only decisions strictly after this timestamp can count:

`2026-09-24T11:33:05.785-03:00`

Equivalent UTC timestamp:

`2026-09-24T14:33:05.785000+00:00`

Any data before this anchor is diagnostic only.

## Why this exists

Fast182 found that the Fast177 entry gate (`edge >= 7.5` plus high ETHUSDT
30-minute quote volume) was negative under fixed terminal exit, but became
positive historically with wider low-cost management:

- validation stress: `+188.4397695578296` bps;
- extension stress: `+135.41918576065552` bps;
- total stress: `+323.8589553184852` bps;
- trades: `207`.

Fast183 freezes that best candidate for future-only observation.

## Frozen entry policy

Source signal:

- target: `ETHUSDT`;
- model, features, training set, decision stride, horizon, and regime are the
  same as Fast177;
- source diagnostic rows must come from the current Fast177 observer output.

Entry rule:

```text
abs(prediction) >= 7.5
ETHUSDT_qv_30m >= 100000000.0
decision_time > 2026-09-24T14:33:05.785000+00:00
```

Direction:

- `LONG` if `prediction > 0`;
- `SHORT` if `prediction < 0`.

## Frozen management policy

Policy: `micro_8_3`.

Exact parameters:

- latency: `100ms`;
- expected round-trip cost: `1.0` bps;
- stress round-trip cost: `2.0` bps;
- initial stop: `16` bps;
- breakeven activation: `8` bps;
- breakeven lock: `2` bps;
- trailing activation: `12` bps;
- trailing distance: `3` bps;
- horizon timeout: `900s`.

The outcome metric for gates is `stress_net_bps`.

## Gates

Early failure:

- at least `20` eligible decisions and total stress net bps `<= -100.0`; or
- any sufficient day with at least `5` eligible decisions and day stress net bps
  `<= -60.0`.

Shadow candidate gate:

- at least `60` eligible decisions;
- at least `5` days;
- stress net mean bps `> 1.0`;
- stress net total bps `> 0`;
- positive sufficient-day share `>= 0.60`;
- one-sided lower confidence bound `> 0`;
- no sufficient-day loss breach.

Even if the shadow candidate gate passes, Fast183 remains offline research until
explicit human review and separate authorization.

## Safety

Fast183 must always report:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

No order submission, no ledger commit, no Demo/Testnet/LIVE activation.
