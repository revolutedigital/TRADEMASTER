# Pre-registration: microstructure research v1

Status: architecture contract approved on 2026-09-21. No market-data partition is
opened by this document and no experiment is frozen until its dataset manifest,
account cost profile, latency profile, and immutable experiment hash are recorded.

## Research question

Can causal market-microstructure information identify BTCUSDT entries whose
event-by-event path has enough favorable excursion to pay the complete entry and
exit cost when managed by an initial stop, net breakeven, and trailing stop?

This is intentionally different from predicting the return at a fixed future
timestamp. The primary target is the executable path after the decision.

## Product contract

- execution venue: Binance USD-M Futures;
- execution instrument: BTCUSDT perpetual;
- supported directions: long and short;
- auxiliary signal venue: Binance Spot BTCUSDT;
- auxiliary information never changes the instrument used for fills and P&L;
- maximum holding horizons: 5, 15, 30, 120, and 300 seconds;
- position model: at most one net BTCUSDT position at a time;
- research stages allowed by this contract: offline replay and shadow only;
- order submission: forbidden;
- Testnet: requires a later, explicit release decision after prospective evidence;
- LIVE: outside this contract and never authorized by research approval.

An approval under this protocol means only that a frozen candidate may enter the
next evidence stage. It is not permission to deploy, activate, arm, or trade.

## Evidence boundary

The event clock is exchange event time, with receive time retained separately.
Executed trades, best bid/ask, depth updates, mark price, funding, and liquidation
events keep their native sequence identifiers. Missing or non-monotonic sequences
are quarantined; they are never silently interpolated.

The first research build uses:

1. at least 90 calendar days of official historical trades or aggregate trades,
   mark price, and funding for the execution instrument;
2. prospectively recorded spot and futures trades plus top-of-book and depth;
3. at least 60 complete days of order-book evidence before a book-dependent model
   can be audited, unless a separately validated historical L2 source is approved;
4. a final 20 to 30 calendar-day prospective block recorded only after the model,
   top-p policy, execution policy, and management rules are frozen.

Raw source files are immutable. Every normalized partition carries its source URL,
venue, product, symbol, UTC interval, checksum, schema version, and quality status.
PostgreSQL stores manifests and lineage; event payloads remain in compressed raw
files and partitioned Parquet.

## Economic feasibility gate

No model work begins for a horizon until an edge-budget report measures:

- favorable and adverse excursion distributions;
- observed spread and trade-size distribution;
- maker and taker commissions supplied for the intended account;
- funding attributable to the holding interval;
- slippage under the intended order style;
- measured or conservatively simulated decision-to-exchange latency.

The account cost profile is mandatory before an experiment can be frozen. The
expected scenario uses that profile. The stress scenario uses at least twice the
expected round-trip friction and never less than 20 bps round trip. A maker claim
is allowed only when queue position, non-fill, partial-fill, and adverse-selection
effects are represented; otherwise the replay assumes a marketable taker order.

A horizon is rejected before machine learning when its available excursion cannot
clear the expected cost with a positive economic margin.

## Outcomes and labels

For every causal decision time and direction, the label factory records:

- maximum favorable excursion (MFE) and maximum adverse excursion (MAE);
- first touch among stop, net breakeven threshold, profit threshold, and timeout;
- time to each touch and event sequence responsible for it;
- realized net result for each pre-registered management policy;
- whether the path paid expected and stress costs before the initial stop;
- fill state, filled quantity, and unfilled remainder.

Stops and trailing changes become effective only after the event that caused the
change. Events sharing a timestamp retain exchange sequence order. No OHLC-based
intrabar assumption may replace an available event sequence.

## Candidate information families

The bounded v1 universe contains only these causal families:

1. trade-flow direction, acceleration, size, and inter-arrival duration;
2. spread, depth imbalance, microprice, and top-of-book pressure;
3. sweep, absorption, replenishment, cancellation, and book recovery;
4. spot-perpetual lead/lag, basis, mark/index displacement, and funding;
5. liquidation flow, volatility, session, and liquidity regime.

Each family must pass point-in-time and offline/online parity tests. A family is
evaluated by ablation. Adding an indicator, interaction, threshold, feature family,
or management rule after observing audit data creates a new experiment and burns
the opened partition for future confirmation.

## Model and top-p policy

The first benchmark is a regularized, calibrated linear or survival model. A tree
model may advance only if it improves untouched validation after the linear
benchmark and the comparison is counted in the global hypothesis ledger.

Top-p is not a fixed raw score. For each training fold and market regime it is the
tail quantile of an out-of-sample calibrated probability that the path clears all
costs before its initial stop. The only v1 selection tails are 1%, 2%, 5%, and 10%.
Thresholds are fitted without access to the subsequent test block. The 1% tail must
not perform worse than 5%, and 5% must not perform worse than 10%, within sampling
uncertainty; otherwise calibration is rejected.

## Temporal separation

Every experiment declares concrete UTC partitions before reading outcomes:

1. development: feature and implementation diagnostics only;
2. training: model fitting;
3. selection: bounded choice among pre-registered candidates;
4. audit: one-time evaluation after every choice is frozen;
5. prospective shadow: 20 to 30 days recorded after the audit candidate is frozen.

Training and evaluation use purged, embargoed, nested walk-forward splits. The
embargo is at least the maximum label horizon plus configured execution latency.
Overlapping labels are not treated as independent observations. Confidence
intervals use time-block resampling, never an IID trade bootstrap.

## Approval gate

A candidate is `APPROVED` for shadow continuation only when all conditions hold:

- at least three traded out-of-sample temporal folds;
- at least 200 out-of-sample portfolio trades on at least 20 distinct UTC days;
- positive net mean and a multiplicity-adjusted one-sided 95% lower confidence
  bound above zero under the expected cost profile;
- positive net mean under the stress cost profile;
- top-p calibration is monotonic as defined above;
- probability of backtest overfitting is at most 20%;
- the same frozen policy remains positive in the prospective shadow block;
- no unresolved data-quality, fill-model, lineage, or reconciliation failure.

Portfolio results, not independent signal rows, are the approval unit. The replay
must enforce available capital, one net position, fill quantities, funding, fees,
slippage, latency, rejected overlaps, and the exact stop/breakeven/trailing policy.

Failure of a necessary condition produces `REJECTED`. Insufficient power or data
produces `INCONCLUSIVE`, never approval. Rejected and inconclusive experiments do
not authorize parameter adjustment against their audit or prospective partitions.

## Global multiplicity and burned-data rules

The experiment registry counts every attempted target, horizon, direction, feature
set, model, top-p tail, cost profile, order style, and management policy, including
failed jobs and manually inspected outputs. Deleting a result does not erase it.

Opening any audit or prospective outcome burns that partition for all descendant
experiments. A descendant may use it for historical comparison, but must acquire a
new untouched confirmation block. Experiment status is one of `DRAFT`, `FROZEN`,
`REJECTED`, `INCONCLUSIVE`, or `APPROVED`; only `DRAFT` is mutable.

## Fail-closed safety boundary

The research API is metadata and evidence only. It exposes no order, deployment,
activation, engine-start, arming, credential, or execution-mode mutation endpoint.
Every response declares `research_only: true`, `order_submission_allowed: false`,
and `execution_authorization: none`.

Existing PAPER, TESTNET, and LIVE controls remain unchanged. Implementing this
protocol must not modify runtime execution settings, strategy deployments, API
credentials, order managers, or the live-trading guard.
