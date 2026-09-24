# Fast185 preregistration — Fast177 edge7.5 + micro_5_2 + UTC 05-16

Preregistered at São Paulo time:

`2026-09-24T11:51:53-03:00`

Equivalent UTC timestamp:

`2026-09-24T14:51:53+00:00`

## Purpose

Fast184 found one historical post-training survivor that was positive on every
full historical day from 2026-09-15 through 2026-09-23. Fast185 freezes that
candidate before any later decision can count.

This is a future-only offline observation. It does not authorize Demo, Testnet,
LIVE, order submission, or ledger writes.

## Frozen rule

Target:

- `ETHUSDT`

Source signal:

- Fast177-style cross-asset lead-lag prediction trained only on the Fast174
  training blocks: `fast50`, `fast51`, `fast52`.

Entry gate:

- `abs(prediction) >= 7.5 bps`;
- `ETHUSDT_qv_30m >= 100,000,000`;
- prediction sign determines side:
  - positive prediction = `LONG`;
  - negative prediction = `SHORT`;
- decision time must be strictly after `2026-09-24T14:51:53+00:00`;
- UTC decision hour must be from `05` through `16`, inclusive.

Management:

- policy: `micro_5_2`;
- initial stop: `12 bps`;
- breakeven activation: `5 bps`;
- breakeven lock: `2 bps`;
- trailing activation: `8 bps`;
- trailing distance: `2 bps`;
- no partial take-profit.

Execution assumptions:

- horizon: `900 seconds`;
- latency: `100 ms`;
- expected round-trip cost: `1.0 bps`;
- stress round-trip cost: `2.0 bps`.

## Observation gates

Fast185 uses the same offline observation gate family as Fast180/Fast183:

- early fail if at least 20 decisions and stress net total is `<= -100 bps`;
- early fail if any sufficient day, defined as at least 5 decisions, is
  `<= -60 bps`;
- shadow candidate only after at least 60 decisions, at least 5 observed days,
  positive stress total, stress mean above `1.0 bps`, positive sufficient-day
  share at least `60%`, and positive lower confidence bound.

Even if the shadow gate passes, the status remains not approved and requires
manual review before any testnet decision.

## Safety boundary

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

## Historical parent evidence

Fast184 selected:

- candidate: `edge7p5_micro5_all_utc05_16`;
- trades: `255`;
- stress total: `+188.05480423610305 bps`;
- positive days: `9/9`;
- worst day: `+1.7416465476528282 bps`.

Parent report:

- `docs/forex/fast184-robust-session-filter-historical-audit-report.md`
- report SHA256:
  `9c0bf03d5035d1f3cea61759134c9a596bbd332fe8e9c47173405bb6c7273af8`
