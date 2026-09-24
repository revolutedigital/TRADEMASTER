# Fast180 — Fast178 edge-2 with micro_3_1 management preregistration

Fast180 is a future-only offline observation derived from Fast178 and the
Fast179 path-management diagnostic. It does not amend Fast178, does not approve
a strategy, and does not authorize Demo, Testnet or LIVE execution.

## Preregistration anchor

Only decisions strictly after this timestamp can count:

`2026-09-24T11:17:46.784-03:00`

Equivalent UTC timestamp:

`2026-09-24T14:17:46.784000+00:00`

Any data before this anchor is diagnostic only.

## Why this exists

Fast178's first eligible edge-2 decision was directionally wrong at the fixed
15-minute terminal horizon, but the Fast179 path audit showed the path first
moved in the predicted direction before reversing. The best diagnostic
management on that one eligible Fast178 row was `micro_3_1`, with positive
stress net bps, but the sample was only one trade and therefore not strategy
evidence.

Fast180 freezes that exact next question before observing future rows:

> If the same Fast178 entry gate appears again, does short breakeven/trailing
> management consistently extract the initial favorable move?

## Frozen entry policy

Source signal:

- target: `ETHUSDT`;
- model, features, training set, decision stride, horizon, and regime are the
  same as Fast178;
- source diagnostic rows must come from the current Fast178 observer output.

Entry rule:

```text
abs(prediction) >= 2.0
ETHUSDT_qv_30m >= 100000000.0
decision_time > 2026-09-24T14:17:46.784000+00:00
```

Direction:

- `LONG` if `prediction > 0`;
- `SHORT` if `prediction < 0`.

## Frozen management policy

Policy: `micro_3_1`.

Exact parameters:

- latency: `100ms`;
- expected round-trip cost: `1.0` bps;
- stress round-trip cost: `2.0` bps;
- initial stop: `8` bps;
- breakeven activation: `3` bps;
- breakeven lock: `2` bps;
- trailing activation: `5` bps;
- trailing distance: `1` bps;
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

Even if the shadow candidate gate passes, Fast180 remains offline research until
explicit human review and separate authorization.

## Safety

Fast180 must always report:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

No order submission, no ledger commit, no Demo/Testnet/LIVE activation.
