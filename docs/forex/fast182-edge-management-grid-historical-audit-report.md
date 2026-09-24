# Fast182 — edge-management grid historical audit

Fast182 widened Fast181 from one management policy to the full low-cost
management grid already in the codebase:

- `micro_2_1`;
- `micro_3_1`;
- `micro_5_2`;
- `micro_8_3`.

Status:

`FAST182_EDGE_MANAGEMENT_GRID_STRICT_HISTORICAL_SURVIVORS`

## Result

Fast182 found `10` historical strict survivors: positive in both Fast174
validation (`fast53`) and extension (`fast56`).

Best candidate:

- edge threshold: `7.5` bps;
- management: `micro_8_3`;
- trades: `207`;
- validation stress: `+188.4397695578296` bps;
- extension stress: `+135.41918576065552` bps;
- total stress: `+323.8589553184852` bps;
- mean stress: `+1.5645360160313293` bps;
- stress win rate: `0.5314009661835749`.

This is the first materially interesting branch in this session: the same entry
family that failed with fixed terminal exit becomes positive historically when
managed with wider low-cost breakeven/trailing.

It is still not a strategy. It is historical selection and requires
future-only preregistration.

## Top candidates

| Edge | Management | Trades | Validation | Extension | Total |
|---:|---|---:|---:|---:|---:|
| `7.5` | `micro_8_3` | `207` | `+188.4397695578296` | `+135.41918576065552` | `+323.8589553184852` |
| `7.5` | `micro_5_2` | `207` | `+87.97622602042665` | `+156.55150936411627` | `+244.52773538454292` |
| `10.0` | `micro_8_3` | `119` | `+136.23863734826642` | `+104.72343176367824` | `+240.96206911194463` |
| `10.0` | `micro_5_2` | `119` | `+67.6329486787125` | `+129.76229839963332` | `+197.39524707834582` |
| `5.0` | `micro_5_2` | `355` | `+68.42513611304649` | `+126.27513314178552` | `+194.700269254832` |

## Safety

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`.

## Artifacts

- runner:
  `backend/scripts/research/run_fast182_edge_management_grid_historical_audit.py`;
- test:
  `backend/tests/unit/test_fast182_edge_management_grid_historical_audit.py`;
- report:
  `backend/data/microstructure_cross_asset_v1/reports/fast182-edge-management-grid-historical-audit/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast182-edge-management-grid-historical-audit/summary.json`.

Hashes:

- report: `83ee0a3dbbd3f2e9b2a29f317476f4555cc9a7f4fe94b9b7d9f93a90f405f9a5`;
- summary: `6dc47069926fdd8e942a81f59afd1d5d427fff02fb7cc0dd3a7a37cee83aeafe`.
