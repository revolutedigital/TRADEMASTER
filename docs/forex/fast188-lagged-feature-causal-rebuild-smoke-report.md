# Fast188 — lagged-feature causal rebuild smoke

Date: 2026-09-24

Status: `FAST188_LAGGED_FEATURE_CAUSAL_REBUILD_NO_SIMPLE_SURVIVOR`

This is not a strategy approval. Fast188 is a conservative causal smoke after
Fast187 rejected Fast185 under a 60-second close-delay stress.

## Question

If every model feature is forced to be one 5-minute decision step old, does the
same cross-asset idea still produce a simple robust threshold/policy candidate?

## Setup

- ETHUSDT target;
- training blocks: `fast50`, `fast51`, `fast52`;
- post-training dates: 2026-09-15 through 2026-09-23;
- UTC session: `05-16`;
- regime gate: `ETHUSDT_qv_30m >= 100,000,000`, also lagged;
- feature lag: one 5-minute decision step;
- policies tested: `micro_3_1`, `micro_5_2`, `micro_8_3`;
- edge thresholds tested: `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `7.5`, `10.0`.

## Result

No simple lagged-feature threshold/policy candidate survived.

Best-looking candidates by positive-day count / total bps were still negative:

| Candidate | Trades | Stress total bps | Mean bps | Positive days | Worst day bps |
|---|---:|---:|---:|---:|---:|
| `lag1_edge10_micro_5_2_all_utc05_16` | `148` | `-134.37557124433403` | `-0.9079430489482029` | `4/9` | `-65.92816979554394` |
| `lag1_edge10_micro_8_3_all_utc05_16` | `148` | `-178.81332250427806` | `-1.2081981250289058` | `4/9` | `-166.49424126551236` |
| `lag1_edge7.5_micro_8_3_all_utc05_16` | `271` | `-304.19095599384474` | `-1.122475852375811` | `4/9` | `-191.52887339752024` |
| `lag1_edge5_micro_5_2_all_utc05_16` | `436` | `-419.3148676468824` | `-0.9617313478139505` | `4/9` | `-194.58373433316598` |

## Interpretation

This confirms the Fast187 warning: the profitable Fast185 line was not robust
to a causal timestamp contract.

Forcing features to be known before the decision does not merely reduce the
edge; it flips the simple threshold/policy family negative. The next research
step should not be another parameter tweak on Fast185. It should rebuild the
dataset contract:

1. define exact feature close time;
2. define exact executable decision time;
3. train labels from that executable decision time;
4. then run discovery again under that causal contract.

## Artifacts

- Report JSON: `backend/data/microstructure_cross_asset_v1/reports/fast188-lagged-feature-causal-rebuild-smoke/report.json`
- Report SHA256: `9aba5211384feaa26e8988413d41a092016f956f423f145be7e7416fe9219ec9`
- Summary SHA256: `7af6e6875cf4b7c82df0d386f96c6ba09bd177eedcfef98b0446dc74e771de59`
- Settlement SHA256: `14579cbe4a5d500dd3e80223516bf7c330cd3b993c65ec978e9ff943214ba1ce`
