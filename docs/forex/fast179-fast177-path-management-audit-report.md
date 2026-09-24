# Fast179 — Fast177/Fast178 path-management audit

Fast179 tested Igor's trailing-stop hypothesis on the current prospective
Fast177/Fast178 rows: did the signal move in our favor first, and would short
breakeven/trailing extract that move?

Status:

`FAST179_PATH_MANAGEMENT_REJECTS_CURRENT_ELIGIBLE_ENTRIES`

## Result

Fast179 initially looked promising on the first Fast178 row, but after the
14:15 UTC update the sample expanded to three Fast178 eligible rows and the
management shortcut failed.

Best eligible policy:

- source: `fast178`;
- policy: `micro_3_1`;
- rows: `3`;
- stress net total: `-3.6802835216880396` bps;
- stress mean: `-1.2267611738960131` bps;
- stress win rate: `0.6666666666666666`;
- exits: `2` breakeven stops, `1` initial stop.

Fast178 eligible rows under terminal stress were worse:

- terminal stress net total: `-53.85745692956904` bps;
- terminal stress win rate: `0.0`;
- only `1/3` rows moved favorably before the adverse threshold.

Interpretation: tight management reduces loss, but does not create a validated
edge. The `13:55Z` short was the failure case: it had edge `8.759740972961417`,
but moved against immediately and hit the initial stop.

## Safety

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

## Artifacts

- runner:
  `backend/scripts/research/run_fast179_fast177_path_management_audit.py`;
- test:
  `backend/tests/unit/test_fast179_fast177_path_management_audit.py`;
- report:
  `backend/data/microstructure_cross_asset_v1/reports/fast179-fast177-path-management-audit/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast179-fast177-path-management-audit/summary.json`.

Hashes:

- report: `67bec565416d79afbdf4a3e502155db244cf20489bb6ce0cd78c8bf74ecb7c5d`;
- summary: `e925159ecf864db349a8bb7436db091f2b32e5504482cba1df5ca4bcdf7fb905`.
