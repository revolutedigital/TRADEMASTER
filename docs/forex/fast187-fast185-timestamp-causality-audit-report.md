# Fast187 — Fast185 timestamp-causality audit

Date: 2026-09-24

Status: `FAST187_REJECTS_FAST185_UNDER_CAUSAL_CLOSE_DELAY`

This is not a strategy approval. Fast187 audits whether Fast185 survives a
causal execution delay that represents waiting for the one-minute feature bar
to be observable before entering.

## Why this matters

The Fast174/Fast177/Fast185 family uses one-minute resampled features. If the
feature row labelled `15:30:00` contains information from the interval
`15:30:00-15:30:59`, an execution system cannot use that information at
`15:30:00.100`. It must either:

- enter after the minute is closed and observable; or
- shift/lag the features so the decision only uses information already known.

Fast187 keeps the Fast185 rule fixed and only delays the simulated entry.

## Fixed candidate tested

- ETHUSDT;
- `abs(prediction) >= 7.5 bps`;
- `ETHUSDT_qv_30m >= 100,000,000`;
- UTC session `05-16`;
- policy `micro_5_2`;
- stress cost `2.0 bps`;
- post-training dates 2026-09-15 through 2026-09-23.

## Result

| Entry delay | Trades | Stress total bps | Mean bps | Positive days | Worst day bps |
|---:|---:|---:|---:|---:|---:|
| `0s` | `255` | `+188.05480423610305` | `+0.7374698205337374` | `9/9` | `+1.7416465476528282` |
| `60s` | `255` | `-280.4483046702661` | `-1.0997972732167298` | `3/9` | `-100.13553546684241` |
| `120s` | `255` | `-532.787457249103` | `-2.0893625774474627` | `1/9` | `-142.2182148048243` |
| `300s` | `255` | `-373.5969271990692` | `-1.4650859890159578` | `2/9` | `-133.5754008352407` |

## Interpretation

The Fast185 result does not survive a realistic close-delay stress.

That means the current candidate should be treated as contaminated by
timestamp/entry alignment risk. This does not prove the whole research direction
is impossible; it says the current edge must be rebuilt with a strictly causal
feature/entry contract before any strategy, testnet, or bot decision.

## Decision

Fast185 remains useful as a research trail, but it is rejected as an executable
strategy candidate until we fix one of these:

1. shift the feature rows so decision time means “bar is already closed”;
2. or shift execution to the next observable timestamp and re-run the full
   discovery/validation pipeline;
3. or rebuild a model from lagged-only features.

No order execution is allowed from the current Fast185 line.

## Artifacts

- Report JSON: `backend/data/microstructure_cross_asset_v1/reports/fast187-fast185-timestamp-causality-audit/report.json`
- Report SHA256: `3a18b1fb9c49ace2000efc3809049352ac4e5084035d919d42d45ae752b9ebf6`
- Summary SHA256: `1460a5f8d783d8921f710b1ebbab3f860bd18e0cad397c4458fcd3efabd18f6b`
- Delay settlements SHA256: `a788c0c1e128008e90af7c8ef640b9b56158a058be93ab09041e3f6cca24a379`
