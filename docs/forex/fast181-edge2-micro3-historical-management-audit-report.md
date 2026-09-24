# Fast181 — historical edge-2 + micro_3_1 management audit

Fast181 tested the Fast178-style ETHUSDT gate historically on the already
defined Fast174 validation/extension blocks, using the same training blocks and
the same `micro_3_1` management from Fast179/Fast180.

Status:

`FAST181_EDGE2_MICRO3_FAILED_EXTENSION`

## Result

The management shortcut improved the fixed terminal outcome, but remained
negative and failed extension.

| Edge threshold | Trades | Validation stress | Extension stress | Total stress |
|---:|---:|---:|---:|---:|
| `2.0` bps | `632` | `-188.74049735367456` | `-137.53366010923438` | `-326.2741574629089` |
| `7.5` bps | `207` | `-30.51564364636044` | `-56.573528033844866` | `-87.08917168020528` |
| `10.0` bps | `119` | `+2.013997616904412` | `-23.19221991935251` | `-21.17822230244809` |

Interpretation:

- trailing/breakeven cuts the terminal loss materially;
- the edge remains negative after costs;
- edge `10` is the closest branch, but it fails extension;
- therefore the current “edge-2/edge-threshold + micro_3_1” family is not a
  strategy and should not be promoted.

## Safety

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

## Artifacts

- runner:
  `backend/scripts/research/run_fast181_edge2_micro3_historical_management_audit.py`;
- test:
  `backend/tests/unit/test_fast181_edge2_micro3_historical_management_audit.py`;
- report:
  `backend/data/microstructure_cross_asset_v1/reports/fast181-edge2-micro3-historical-management-audit/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast181-edge2-micro3-historical-management-audit/summary.json`.

Hashes:

- report: `be614026293adeda4b032d9a4aab3b9205031f93be895ceeef72fbaaacfda145`;
- summary: `50a3c07dc6b5db5d4d087d2b59b15e5fc0289a8373b76de6fd9a03e69561c344`.
