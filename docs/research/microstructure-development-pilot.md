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
reconstructed best bid and ask. That three-stream sample projected roughly
35 GiB for 60 days before `spot_trade` became mandatory; the four-stream
collector must be remeasured after deployment, and a 50 GB volume must not be
treated as proven sufficient for the eligible 60-day window until the spot stream
rate is observed. Binance's zero-price, zero-quantity `trade` heartbeat is
explicitly discarded and is not treated as a market trade.

### Prospective collection clock

The isolated `microstructure-recorder` Railway service began writing to its
50 GB persistent volume in Singapore at `2026-09-22T03:09:51Z`. No API key is
present and the process has no order-submission path. The first durability check
observed the original three-stream command growing on the mounted volume:

| Stream | Initial gzip size | Size 25s later |
| --- | ---: | ---: |
| futures trades | 50,975 bytes | 83,829 bytes |
| reconstructed depth | 385,539 bytes | 631,237 bytes |
| mark price | 11,688 bytes | 20,279 bytes |

After the evidence gate was tightened to require `spot_trade` alongside futures
`trade`, `depth`, and `mark_price`, the production recorder command was updated
to start with `--include-spot-trades`. The earlier three-stream clock is
durability evidence only; it is not an eligible four-stream evidence window. The
60-complete-day gate starts only after the deployed recorder is running with all
four required streams and daily audits validate complete UTC days. Starting or
restarting the recorder does not approve a model or activate Testnet.

On 2026-09-22, production deployment
`970ae4de-6ebe-4088-a2e0-08d8c11eb855` restored the REST premium-index
`mark_price` fallback while keeping the attempted WebSocket subscription and the
mandatory `--include-spot-trades` flag. A live volume check confirmed all four
required WAL streams existed and were still growing:

| Stream | First read | Second read |
| --- | ---: | ---: |
| spot_trade | 296,403 bytes | 300,038 bytes |
| mark_price | 2,916,188 bytes | 2,917,842 bytes |
| depth | 71,905,892 bytes | 71,942,208 bytes |
| trade | 11,457,551 bytes | 11,463,703 bytes |

The same production check later observed the latest `mark_price` row coming from
`rest_premium_index`, confirming the fallback is active. The 2026-09-22 UTC day
remains intentionally ineligible because collection started after midnight UTC,
`spot_trade` started only after the four-stream redeploy, and the mark-price
stream had a probe gap before the fallback was restored. The first possible
eligible day is therefore the first full UTC day after the stable four-stream
deployment, subject to the daily audit passing without gaps.

Recorder reconnects are not allowed to silently stitch two unrelated order-book
sequences together. From the next deployment onward, each futures depth resync
writes an immutable `DEPTH` snapshot boundary (`payload.kind=depth_snapshot`)
with the REST `lastUpdateId`, top-of-book, and snapshot levels before subsequent
depth deltas are accepted. The WAL audit treats the next delta as continuous only
when it bridges that snapshot `lastUpdateId`; an unmarked sequence discontinuity
still invalidates the day. This keeps normal resyncs auditable without weakening
the gap detector.

Deployment `76bfc125-e459-4ffb-96c4-ef84abf12d1f` put that boundary marker in
production on 2026-09-22. The volume check immediately after deploy found
`depth_snapshot_count=1`, with the last snapshot received at
`2026-09-22T06:42:33.896643Z` and `last_update_id=11624103278588`. A second
read confirmed all four required WAL streams were still growing:
`spot_trade` 831,349 → 850,072 bytes, `mark_price` 3,108,822 → 3,112,976 bytes,
`depth` 76,292,216 → 76,411,776 bytes, and `trade` 12,156,982 → 12,182,106
bytes.

A repeatable WAL audit was added at
`backend/scripts/research/audit_microstructure_wal.py`. It verifies, per UTC day,
that the required `trade`, `spot_trade`, `depth`, and `mark_price` gzip JSONL
streams exist, cover the full day boundary, satisfy minimum row counts, stay
within receive-gap tolerances, carry the expected product/event type, and have no
JSON errors, duplicate sequence IDs, sequence regressions, or depth sequence
gaps. `liquidation` remains optional because a quiet day can have zero
forced-order events. The audit output carries a deterministic manifest hash and
repeats the safety boundary: research only, no order submission, no execution
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

### Book/aux-aware dataset path

The historical development result above was intentionally trade-flow-only because
Binance aggregate-trade archives do not contain executable top-of-book state. The
research dataset builder now has a separate path for the prospective WAL depth
stream: it can join causal top-of-book features at each decision time using only
the latest quote observed at or before that decision.

Book-dependent materialization must be run fail-closed:

```bash
cd backend
./.venv/bin/python scripts/research/normalize_prospective_trade_wal.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --wal-root data/microstructure_v1/prospective-wal \
  --output-root data/microstructure_v1/normalized/trades

./.venv/bin/python scripts/research/build_microstructure_dataset.py \
  --market spot \
  --kinds aggTrades \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --root data/microstructure_v1

./.venv/bin/python scripts/research/build_research_dataset.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --source-root data/microstructure_v1/normalized/trades \
  --book-source-root data/microstructure_v1/prospective-wal/depth \
  --mark-source-root data/microstructure_v1/prospective-wal/mark_price \
  --liquidation-source-root data/microstructure_v1/prospective-wal/liquidation \
  --spot-source-root data/microstructure_v1/normalized/spot-aggTrades \
  --require-book-features \
  --max-book-staleness-ms 1000
```

The first command converts recorder `trade/date=*/events.jsonl.gz` WAL files
into the same immutable parquet schema used by the replay. The second command
downloads and normalizes Binance Spot public archive `aggTrades` into
`normalized/spot-aggTrades`; spot timestamps are preserved at the archive's
millisecond/microsecond precision. For fully prospective auxiliary evidence, run
the recorder with `--include-spot-trades`; it listens only to the public Spot
trade stream and writes those events under
`prospective-wal/spot_trade/date=*/events.jsonl.gz`, which can be passed as
`--spot-source-root` instead of the historical spot archive. The third command
adds `book_available`,
`book_update_age_ms`, `spread_bps`,
`depth_imbalance`, `microprice_displacement_bps`, and short-window top-of-book
dynamics: event count, bid/ask replenishment, bid/ask removed liquidity, net book
pressure, spread widening/recovery, depth-imbalance change, and microprice-change
proxies. Directional versions are emitted for imbalance, microprice displacement,
book pressure, depth-imbalance change, and microprice-change features. Optional
mark/liquidation joins add mark availability, mark/index basis, funding, and
windowed liquidation count/quantity/notional with directional liquidation
features. Optional spot trade joins add causal Binance Spot-vs-USD-M features:
spot trade count/quote volume/flow/return/volatility/interarrival, latest
spot-update age, spot-perp basis, return gap, flow gap, quote-volume ratio, and
directional spot/perp gap columns. The spot source is an auxiliary signal venue
only; execution labels, replay, costs, and P&L stay bound to the futures
`--source-root`. If any decision lacks fresh book state, the partition fails
instead of silently producing a fake “book” model.

The top-p pilot recognizes book-specific feature families only when those columns
exist (`flow_book`, `flow_price_book`, `flow_price_book_session`). It also adds
auxiliary sets (`flow_aux`, `flow_price_aux`, `flow_price_aux_session`, and
book+aux variants) only when real mark/liquidation/spot-perp columns exist.
Without real book or auxiliary columns it continues to run only the historical
flow/price/session families, so a missing WAL join cannot be mistaken for a
book-edge, liquidation/funding-edge, or spot/perp lead-lag test.

Once a candidate is selected, the probability rule used for prospective shadow
must be frozen into a deterministic artifact before any shadow partition opens:

```bash
cd backend
./.venv/bin/python scripts/research/freeze_top_p_policy.py \
  --dataset-root data/microstructure_v1/research-v1 \
  --output data/microstructure_v1/models/frozen-shadow-policy.json \
  --target stress \
  --horizon-seconds 300 \
  --feature-set flow_price_book_aux_session \
  --tail-fraction 0.05 \
  --calibration-date YYYY-MM-DD
```

The artifact is JSON-only and includes feature columns, standardization values,
logistic coefficients, sigmoid calibration parameters, the top-p probability
threshold, the dataset fingerprint, safety flags, and its own `model_sha256`.
The dataset fingerprint includes every `research_rows.parquet` file and its
paired `research_rows.manifest.json`, so changing source lineage, feature
families, or rows changes the frozen policy input hash. That model hash is the
value that shadow signals should record as `model_sha256`. The artifact is still research-only:
`order_submission_allowed=false`, `execution_authorization=none`.

Prospective shadow runners should score and record decisions through
`app.services.research.shadow_policy_runner.record_frozen_top_p_shadow_signal`.
That function verifies the artifact hash, computes the probability from the
serialized coefficients, uses the artifact threshold and horizon, and appends the
signal through the immutable shadow recorder. It has no exchange adapter and no
order-submission path.

For partition batches, use the dry-run-first CLI. Without `--commit`, it only
scores the frozen policy and prints the selected shadow entries:

```bash
cd backend
./.venv/bin/python scripts/research/run_shadow_policy_batch.py \
  --experiment-id EXPERIMENT_ID \
  --policy-artifact data/microstructure_v1/models/frozen-shadow-policy.json \
  --dataset-root data/microstructure_v1/research-v1 \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD
```

Only after the experiment is `FROZEN` and the `PROSPECTIVE_SHADOW` partition has
been explicitly opened should the same command be re-run with `--commit`. The
batch records entry candidates where the frozen probability clears the frozen
top-p threshold, skips already-recorded `(decision_time, side, horizon)` signals,
and still returns `order_submission_allowed=false` and
`execution_authorization=none`.

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

The statistical gate must read the top-p pilot report instead of assuming
calibration passed:

```bash
cd backend
./.venv/bin/python scripts/research/evaluate_statistical_gate.py \
  --portfolio-root data/microstructure_v1/reports/trailing-pilot-stress-target \
  --top-p-report data/microstructure_v1/reports/top-p-pilot-stress.json \
  --prospective-shadow-report data/microstructure_v1/reports/prospective-shadow-settlement.json
```

If the top-p report is missing, lacks research safety flags
(`research_only=true`, `order_submission_allowed=false`,
`execution_authorization=none`), or any result lacks a positive monotonicity
check, or if the prospective shadow report is missing, dry-run, incomplete, or
non-positive under stress costs, the gate records the concrete reason and fails
closed.

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
append-only shadow ledger. It is still metadata only: order submission remains
blocked and execution authorization remains `none`. A 20-to-30-day shadow block
counts toward Testnet only when every shadow signal has immutable outcome
evidence and both expected and stress mean bps are positive. Outcomes are
recorded once through the research shadow recorder with `expected_net_bps`,
`stress_net_bps`, and `label_sha256`; overwrite, missing, malformed, or
non-finite `outcome_json` is fail-closed.

The separate
`POST /api/v1/research/microstructure/experiments/{experiment_id}/testnet-release`
endpoint records the explicit manual research release only after the experiment
is already `APPROVED`, the 60-day book gate passes, the prospective shadow block
is complete and positive, and there are no unresolved evidence failures. It is
idempotent and stores a snapshot hashable release record; it still does not
activate Testnet, start a strategy, load credentials, or submit orders. After
that record exists, the eligibility checklist may return
`explicit_testnet_release=true` and `release_request_required=false`, while
`order_submission_allowed=false` and `execution_authorization=none` remain true.

The experiment report endpoint,
`GET /api/v1/research/microstructure/experiments/{experiment_id}/report`, now
returns concrete report metrics instead of an empty placeholder: experiment hash,
book-evidence availability/streak/reasons, shadow signal/outcome counts,
expected/stress mean bps, shadow completeness/positivity, and the Testnet boundary
flags. It also emits a deterministic `artifact_sha256` for that report payload.

Once an offline/statistical gate has produced its terminal result, the decision
can be recorded through
`POST /api/v1/research/microstructure/experiments/{experiment_id}/decision` with
`REJECTED`, `INCONCLUSIVE`, or `APPROVED` and concrete reasons. This is a
research-ledger mutation only. Even an `APPROVED` research decision still leaves
`order_submission_allowed=false`, `execution_authorization=none`, and Testnet
blocked until the separate evidence and explicit-release gates pass.

Shadow runners can append evidence through the research API:

- `POST /api/v1/research/microstructure/experiments/{experiment_id}/partitions/PROSPECTIVE_SHADOW/open`
  marks the first access to the preregistered shadow block. This operation is
  idempotent but irreversible in the research ledger.
- `POST /api/v1/research/microstructure/experiments/{experiment_id}/shadow-signals`
  records a hypothetical decision after the `PROSPECTIVE_SHADOW` partition is
  explicitly opened. It stores `feature_vector_sha256`, not raw feature values.
- `POST /api/v1/research/microstructure/shadow-signals/{signal_id}/outcome`
  records the one-shot outcome for that signal.

Both endpoints are metadata-only and return
`order_submission_allowed=false` and `execution_authorization=none`.

When the complete event path for a shadow signal is available, settlement should
be produced by `app.services.research.shadow_outcome_settlement`: it replays the
same trailing policy through `HistoricalTrailingSimulator`, returns
`expected_net_bps`, `stress_net_bps`, and a deterministic `label_sha256`, and can
be passed directly to the immutable shadow-outcome recorder. For live shadow
bookkeeping, use `settle_pending_shadow_outcomes(...)`: it selects only pending
signals whose full horizon has matured, records one immutable replay outcome per
signal, and leaves immature signals untouched. It does not call an exchange and
does not create orders.

The matching batch CLI is also dry-run by default:

```bash
cd backend
./.venv/bin/python scripts/research/settle_shadow_outcomes.py \
  --experiment-id EXPERIMENT_ID \
  --trade-root data/microstructure_v1/normalized/aggTrades \
  --policy-name wide
```

Re-run with `--commit` only to append immutable replay outcomes for already
matured shadow signals. The policy name is explicit because changing trailing
management changes P&L; the command refuses unknown policies and still has no
exchange, Testnet, LIVE, credential, or order-submission path.
