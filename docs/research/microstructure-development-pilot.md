# Microstructure v1 — development pilot evidence

Generated from the bounded development block opened on 2026-06-30 through
2026-07-06. This document is research evidence, not an execution authorization.

## Safety status

- research only: **true**;
- order submission allowed: **false**;
- execution authorization: **none**;
- Testnet activated: **no**;
- LIVE activated: **no**.

## Data evidence

The official Binance USD-M archive was downloaded and normalized for 90 complete
UTC days, 2026-06-23 through 2026-09-20:

- 107,332,065 BTCUSDT aggregate trades;
- 90 daily partitions, all `VALID`;
- zero aggregate-trade sequence gaps and zero duplicate aggregate IDs;
- 270 funding observations;
- 129,601 one-minute mark-price observations;
- partition-manifest hash:
  `b9c1ad67819e67b870afb77e23525acefe1d93bb8fc8fa999982861fd0867383`.

The prospective recorder was smoke-tested against the live public feed. In a
clean 30-second run it crash-durably recorded 1,092 raw trades, 283 depth updates,
and 24 mark-price observations in 204 KiB of gzip WAL. Redundant book-ticker
events are not subscribed because every validated depth event already carries the
reconstructed best bid and ask. The observed storage rate projects to roughly
35 GiB for 60 days. Binance's zero-price, zero-quantity `trade` heartbeat is
explicitly discarded and is not treated as a market trade.

### Prospective collection clock

The isolated `microstructure-recorder` Railway service began writing to its
50 GB persistent volume in Singapore at `2026-09-22T03:09:51Z`. No API key is
present and the process has no order-submission path. The first durability check
observed all three files growing on the mounted volume:

| Stream | Initial gzip size | Size 25s later |
| --- | ---: | ---: |
| futures trades | 50,975 bytes | 83,829 bytes |
| reconstructed depth | 385,539 bytes | 631,237 bytes |
| mark price | 11,688 bytes | 20,279 bytes |

The earliest eligible end of the pre-registered 60-complete-day book evidence
window is `2026-11-21T03:09:51Z`, subject to daily completeness and sequence-gap
validation. Starting the recorder does not approve a model or activate Testnet.

A repeatable WAL audit was added at
`backend/scripts/research/audit_microstructure_wal.py`. It verifies, per UTC day,
that the required `trade`, `depth`, and `mark_price` gzip JSONL streams exist,
cover the full day boundary, satisfy minimum row counts, stay within receive-gap
tolerances, and have no JSON errors, duplicate sequence IDs, sequence regressions,
or depth sequence gaps. `liquidation` remains optional because a quiet day can have
zero forced-order events. The audit output carries a deterministic manifest hash
and repeats the safety boundary: research only, no order submission, no execution
authorization.

The same CLI now also emits a book-evidence gate. That gate requires a contiguous
60-complete-day window before any book-dependent audit can be considered eligible.
Non-contiguous complete days do not pass the gate, and this eligibility remains
metadata only: it does not approve Testnet, does not activate a strategy, and does
not authorize order submission.

A first audit copy was taken from the Railway volume on 2026-09-22 while the day
was still in progress. The result was correctly `PARTIAL`: 75,850 trades, 12,096
depth updates, and 1,128 mark-price rows from approximately `03:09:51Z` through
`03:30:34Z`, with zero JSON errors, zero duplicate sequences, and zero sequence
gaps. The book-evidence gate was therefore false, with zero complete days and
zero contiguous complete days. Daily manifest hash:
`44945a8793ab35114937c3d6be300cd386054e2fcec248464b8324fc2d333863`. Book-gate
manifest hash: `72f0a4b739735b9d5cc1f359f594c69515bc4d95dfe2fb4a788d0ffd2c5aef7e`.

## Economic feasibility

At the expected 12 bps and stress 24 bps round-trip costs, the direction-selecting
oracle showed that 5s, 15s, and 30s horizons do not have enough upper-bound movement
to continue. Only 120s and 300s passed the permissive p99 feasibility screen:

| Horizon | Oracle MFE p99 | Net of 24 bps stress |
| --- | ---: | ---: |
| 5s | 5.85 bps | -18.15 bps |
| 15s | 10.15 bps | -13.85 bps |
| 30s | 14.39 bps | -9.61 bps |
| 120s | 30.78 bps | +6.78 bps |
| 300s | 50.35 bps | +26.35 bps |

This oracle sees the future and may reject a horizon, but cannot approve a signal.

## Probability model result

Seven development days produced 483,812 side/horizon rows. Three purged temporal
out-of-sample folds were used. Thresholds for top 1%, 2%, 5%, and 10% were fitted
on the preceding calibration day, never on each test day.

The regularized calibrated logistic model did find real ranking information:

| Target | Horizon | Best feature family | Mean ROC AUC | Top-1% path success |
| --- | ---: | --- | ---: | ---: |
| clear 12 bps before -20 bps | 120s | flow + price + session | 0.751 | 38.3% |
| clear 12 bps before -20 bps | 300s | flow + price + session | 0.656 | 53.5% |
| clear 24 bps before -20 bps | 120s | flow + price + session | 0.838 | 12.6% |
| clear 24 bps before -20 bps | 300s | flow + price + session | 0.731 | 26.0% |

This is useful classification, but not yet a profitable trading edge. Calibration
also drifted across days: a threshold calibrated as top 1% selected more than 1%
on later test days. The selection policy therefore needs regime-aware prospective
calibration rather than a permanent raw-probability cutoff.

## Portfolio replay result

The replay enforced 100 ms latency, one net position, true ordered trade paths,
initial stops, breakeven changes effective only after the causing event, trailing
stops, timeouts, and 12/24 bps expected/stress round-trip costs. It tested:

- two probability targets;
- 120s and 300s horizons;
- three feature families;
- top 1%, 2%, 5%, and 10%;
- tight, balanced, and wide trailing;
- 50% partial realization at +24 bps;
- fixed realization at +24 bps.

The initial 12 bps target produced 12 portfolio variants. The more demanding
24 bps target produced 20 variants. All 20 final variants were statistically
`REJECTED`. The least-negative final variant was wide trailing at top 5%:

- 297 trades across 3 development days;
- expected mean: **-10.75 bps/trade**;
- stress mean: **-22.75 bps/trade**;
- expected win rate: 14.8%;
- expected cumulative log return: -3,191 bps.

## What was wrong

The earlier intuition conflated three different events:

1. the market moves far enough at some point;
2. the model can identify that path before it happens;
3. the management policy actually retains more than the complete round-trip cost.

The model did improve event 2 substantially. It still did not improve it enough
to overcome event 3. Many selected paths moved favorably, then expired near entry;
the gross mean stayed close to zero and the 12 bps round-trip friction became the
net loss. A stop at the entry price is not financial breakeven: after fees and
slippage it realizes approximately -12 bps.

## Decision and next evidence gate

The trade-flow-only candidate is rejected. It must not be extended into an audit,
shadow signal, Testnet, or LIVE candidate.

The research program itself continues with the prospectively recorded information
that historical aggregate trades do not contain: spread, depth imbalance,
microprice, replenishment, cancellation, sweeps, liquidation flow, and measured
receive latency. A book-dependent model needs 60 contiguous complete days before
audit, and Testnet metadata eligibility now also requires that contiguous
book-evidence window plus a positive 20-to-30-day prospective shadow block,
unresolved failure count of zero, and a later explicit Testnet release. Any new
model or management rule is a new counted hypothesis and may not reuse an opened
audit/prospective block as fresh confirmation.

## Evidence-gate artifact for the panel

The dashboard does not scan raw WAL gzip files. The WAL auditor must produce a
small status artifact after each offline audit:

```bash
cd backend
./.venv/bin/python scripts/research/audit_microstructure_wal.py \
  --root data/microstructure_v1/prospective-wal \
  --rolling-days 60 \
  --write-status \
  --format json
```

Default artifact path:
`backend/data/microstructure_v1/prospective-audits/evidence-gate-status.json`.
The API endpoint `GET /api/v1/research/microstructure/evidence-gate` reads only
that artifact. Missing, invalid, or incomplete evidence is fail-closed:
`artifact_available=false`, `eligible=false`, `order_submission_allowed=false`,
and `execution_authorization=none`.

The research panel also exposes
`GET /api/v1/research/microstructure/experiments/{experiment_id}/testnet-eligibility`.
That endpoint crosses experiment status, the book-evidence artifact, and the
append-only shadow ledger. It is still metadata only: `explicit_testnet_release`
is always false in this read path, `release_request_required=true`, and order
submission remains blocked. A 20-to-30-day shadow block counts toward Testnet
only when every shadow signal has immutable outcome evidence and both expected
and stress mean bps are positive. Outcomes are recorded once through the research
shadow recorder with `expected_net_bps`, `stress_net_bps`, and `label_sha256`;
overwrite, missing, malformed, or non-finite `outcome_json` is fail-closed.
