# Fast184 — robust session-filter historical audit

Date: 2026-09-24

Status: `FAST184_ROBUST_SESSION_FILTER_HISTORICAL_SURVIVOR_REQUIRES_PREREGISTRATION`

This is not a strategy approval. Fast184 is a historical audit that freezes the
best session-filter diagnostic found after Fast182/Fast183. It never sends
orders, never writes a ledger, and keeps `approved_strategy=false`.

## Question

Can the Fast177-style ETHUSDT signal become daily-robust if we keep the same
edge/regime idea but restrict it to a simple UTC session and a tighter trailing
policy?

Training remains the Fast174 training set:

- `fast50`;
- `fast51`;
- `fast52`.

The historical post-training audit set is:

- 2026-09-15 through 2026-09-23, inclusive.

## Selected survivor

| Candidate | Edge | Policy | Side | UTC hours | Trades | Stress total bps | Stress mean bps | Win rate | Positive days | Worst day bps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `edge7p5_micro5_all_utc05_16` | `>=7.5` | `micro_5_2` | all | `05-16` | `255` | `+188.05480423610305` | `+0.7374698205337374` | `52.55%` | `9/9` | `+1.7416465476528282` |

Daily breakdown:

| Date | Trades | Stress total bps |
|---|---:|---:|
| 2026-09-15 | `30` | `+5.982298352989457` |
| 2026-09-16 | `22` | `+1.7416465476528282` |
| 2026-09-17 | `14` | `+46.03642027409624` |
| 2026-09-18 | `49` | `+1.7980889297488485` |
| 2026-09-19 | `20` | `+4.447768334950129` |
| 2026-09-20 | `10` | `+55.39338981553089` |
| 2026-09-21 | `57` | `+56.07207635805911` |
| 2026-09-22 | `26` | `+13.831742562389238` |
| 2026-09-23 | `27` | `+2.751373060686338` |

Exit mix:

- `BREAKEVEN_STOP`: 192;
- `INITIAL_STOP`: 58;
- `TRAILING_STOP`: 4;
- `TIMEOUT`: 1.

## Why this is better than the broad Fast183 candidate

The broad `edge>=7.5 + micro_8_3 + all hours` benchmark produced more total
historical bps, but it was not daily-robust:

| Candidate | Trades | Stress total bps | Positive days | Worst day bps |
|---|---:|---:|---:|---:|
| `edge7p5_micro8_all_utc00_23` | `431` | `+467.9537051573367` | `7/9` | `-72.07622140788763` |

That is exactly the failure mode Igor was worried about: the model can show a
large historical total while still having daily clusters that would be hard to
operate. Fast184 chooses the lower-total but steadier candidate.

## Search result

The robust search scanned:

- edge thresholds: `5.0`, `7.5`, `10.0`, `12.5`, `15.0`;
- policies: `micro_2_1`, `micro_3_1`, `micro_5_2`, `micro_8_3`;
- sides: all, long-only, short-only;
- UTC hour windows from `00-00` through `00-23` / `23-23`.

Robust survivor rule:

- all 9 dates must have at least 2 trades;
- at least 8 of 9 days must be positive.

Results:

- robust survivors: `164`;
- perfect 9/9 positive-day survivors: `2`.

The two perfect-day survivors were:

| Candidate shape | Trades | Stress total bps | Positive days | Worst day bps | Min daily trades |
|---|---:|---:|---:|---:|---:|
| edge `>=7.5`, `micro_5_2`, all sides, UTC `05-16` | `255` | `+188.05480423610305` | `9/9` | `+1.7416465476528282` | `10` |
| edge `>=10.0`, `micro_5_2`, all sides, UTC `01-11` | `80` | `+102.70948517198563` | `9/9` | `+0.14686976395176687` | `3` |

Fast184 selects the first survivor because it has a larger sample, stronger
minimum daily trade count, and materially better worst-day buffer.

## Interpretation

We do have a concrete strategy candidate now:

1. only ETHUSDT;
2. Fast177-style cross-asset prediction;
3. `abs(prediction) >= 7.5 bps`;
4. `ETHUSDT_qv_30m >= 100,000,000`;
5. UTC decision hour from 05 through 16;
6. management policy `micro_5_2`;
7. stress cost `2.0 bps`, expected cost `1.0 bps`, latency `100 ms`;
8. 15-minute horizon.

But this is still historical. The candidate was selected using the historical
post-training period, so it must be preregistered and observed on future-only
rows before it can become a shadow/testnet candidate.

## Artifacts

- Report JSON: `backend/data/microstructure_cross_asset_v1/reports/fast184-robust-session-filter-historical-audit/report.json`
- Report SHA256: `9c0bf03d5035d1f3cea61759134c9a596bbd332fe8e9c47173405bb6c7273af8`
- Summary SHA256: `5dff7093680b76af62cd0b78c27e150aff9f6cf86544d562d1d94f1e0b666e19`
- Selected settlements SHA256: `a38a6fcaf3c7991c334e64922770f0ae109086e5b6e2188654a502cab83ab54a`
- Robust survivors SHA256: `6dce5657955c0ab9d8be91d2f766315fede0584b3124eee95d8267bd9a0749ee`
