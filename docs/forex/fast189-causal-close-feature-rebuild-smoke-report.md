# Fast189 — causal close-time feature rebuild smoke

Date: 2026-09-24

Status: `FAST189_CAUSAL_CLOSE_FEATURE_NO_SIMPLE_SURVIVORS`

This is not a strategy approval. Fast189 is a stricter causal rebuild after
Fast187 rejected the Fast185 candidate under a 60-second feature-close delay,
and Fast188 found no simple survivor when all features were lagged one full
5-minute step.

## Question

If the model is trained and tested from the first executable time after each
one-minute feature bin closes, does the same cross-asset lead/lag family still
produce a robust short-trade candidate?

## Setup

- target: `ETHUSDT`;
- training blocks: `fast50`, `fast51`, `fast52`;
- post-training dates: 2026-09-15 through 2026-09-23;
- feature close delay: `60s`;
- event-entry latency: `100ms`;
- horizon: `900s`;
- UTC session search: `05-16`;
- regime gate: `ETHUSDT_qv_30m >= 100,000,000`;
- policies tested: `micro_3_1`, `micro_5_2`, `micro_8_3`;
- edge thresholds tested: `1.0`, `2.0`, `3.0`, `4.0`, `5.0`, `7.5`, `10.0`, `12.5`, `15.0`;
- side filters tested: `ALL`, `LONG`, `SHORT`.

## Result

No robust survivor was found.

| Metric | Value |
|---|---:|
| Training rows | `1692` |
| Post-training scored rows | `2538` |
| Settlement rows | `7644` |
| Robust survivors | `0` |
| Perfect-day survivors | `0` |

The closest near-misses still failed daily robustness:

| Candidate | Trades | Stress total bps | Mean bps | Positive days | Worst day bps | Min daily trades |
|---|---:|---:|---:|---:|---:|---:|
| `edge10_micro_8_3_LONG_utc14_16` | `34` | `52.23156285521592` | `1.5362224369181152` | `7/9` | `-19.361751163232608` | `2` |
| `edge7.5_micro_8_3_LONG_utc14_16` | `57` | `92.93584927763635` | `1.6304534960988832` | `7/9` | `-28.244881557254196` | `4` |
| `edge7.5_micro_8_3_LONG_utc15_16` | `43` | `71.11010918289767` | `1.6537234693697134` | `6/9` | `-13.403616076777212` | `2` |
| `edge10_micro_5_2_LONG_utc12_16` | `52` | `22.58166888417211` | `0.43426286315715595` | `6/9` | `-28.142497140418108` | `3` |

These are not acceptable for an unattended bot. A positive total over nine days
with losing days still present can be a window-selection artifact, not a
tradable edge.

## Interpretation

Fast189 confirms the main diagnosis from Fast187/Fast188: the earlier Fast184 /
Fast185 family was using timing that looked profitable historically, but the
edge does not survive once the model can only act after the feature bar is
actually observable.

The failure mode is not just “fees killed it.” The larger issue is causal
alignment:

1. the previous features were timestamped at the minute boundary;
2. the trade decision was effectively evaluated as if the information was
   known too early;
3. once labels and entries start from the executable post-close time, no simple
   threshold / side / trailing-policy combination remains robust.

The next valid research step is not another Fast185 parameter tweak. The next
step must rebuild the dataset contract first:

1. define exact feature-open, feature-close, and executable-decision timestamps;
2. generate labels only from executable-decision time plus latency;
3. test features that are explicitly known at decision time;
4. then run discovery / replication / prospective confirmation under that
   contract.

## Safety state

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

## Artifacts

- Script: `backend/scripts/research/run_fast189_causal_close_feature_rebuild_smoke.py`
- Report JSON: `backend/data/microstructure_cross_asset_v1/reports/fast189-causal-close-feature-rebuild-smoke/report.json`
- Report SHA256: `d68d263c529b64ea441bdf2d608747c2dc409fab0a4ddc619d20b36328863f96`
- Summary SHA256: `31dea931eb7b4a397dabc86b134d75c29285f64330640c7af4384642009e7eaa`
- Causal scored rows SHA256: `da286ff65db2da8ada61b4da146ca65aeb379c8bfae14d14e6268d5f9b6ad3ff`
- Causal settlements SHA256: `719e078638b3ad40b1f25373c55af9b4af595aabc5754613e78dcfc940f98179`
- Robust survivors SHA256: `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`
