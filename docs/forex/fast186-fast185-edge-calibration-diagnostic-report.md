# Fast186 — Fast185 edge-calibration diagnostic

Date: 2026-09-24

Status: `FAST186_REJECTS_SIMPLE_EDGE_LOWERING`

This is not a strategy approval. Fast186 checks whether the current Fast185
candidate should simply lower its `edge>=7.5` gate after the first live rows
showed no eligible entries.

## Result

Do not lower the Fast185 edge gate by itself.

The current live rows make the lower edge tempting because the model has seen
some correct low-confidence direction calls. But the historical post-training
audit says the lower thresholds are not daily-robust, and the current live
managed outcome would still be negative under the lower thresholds.

## Historical threshold calibration

Same setup as Fast185:

- ETHUSDT only;
- UTC session `05-16`;
- regime gate `ETHUSDT_qv_30m >= 100,000,000`;
- management `micro_5_2`;
- stress cost `2.0 bps`;
- 15-minute horizon;
- post-training historical dates 2026-09-15 through 2026-09-23.

| Edge threshold | Trades | Stress total bps | Mean bps | Positive days | Worst day bps | Min daily trades |
|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | `1022` | `-12.229863124861616` | `-0.011966597969531914` | `4/9` | `-108.09657237103377` | `71` |
| `0.5` | `957` | `-12.268754015550641` | `-0.012820014645298475` | `5/9` | `-104.932859063689` | `66` |
| `1.0` | `886` | `+26.213179753502487` | `+0.029585981663095357` | `4/9` | `-129.7882714848213` | `59` |
| `1.5` | `820` | `+37.13055147965101` | `+0.04528116034103782` | `5/9` | `-122.45917799160483` | `55` |
| `2.0` | `759` | `+53.404372641680524` | `+0.07036149228152902` | `7/9` | `-133.5114432734884` | `52` |
| `2.5` | `695` | `+136.6526997230225` | `+0.19662258952952877` | `6/9` | `-150.55622963944006` | `47` |
| `3.0` | `633` | `+128.9815152486671` | `+0.20376226737546146` | `6/9` | `-112.66406783742325` | `45` |
| `3.5` | `576` | `+208.59753937632942` | `+0.362148505861683` | `6/9` | `-75.08863198036275` | `38` |
| `4.0` | `530` | `+223.5637359766688` | `+0.42181836976729964` | `6/9` | `-56.05248247965783` | `33` |
| `5.0` | `424` | `+167.42134786297206` | `+0.39486166948814166` | `6/9` | `-46.32449154163097` | `24` |
| `7.5` | `255` | `+188.05480423610305` | `+0.7374698205337374` | `9/9` | `+1.7416465476528282` | `10` |

Interpretation:

- thresholds below `7.5` can have positive total bps;
- but none below `7.5` preserves `9/9` positive days;
- the lower thresholds reintroduce bad-day clusters from `-46 bps` to
  `-150 bps`;
- the selected `7.5` threshold is still the only tested threshold with every
  post-training day positive.

## Current Fast185 live rows

The current Fast185 snapshot has:

- live rows: `8`;
- max edge: `3.622382067242829`;
- eligible Fast185 trades at edge `7.5`: `0`.

If we had lowered the threshold on the current rows:

| Edge threshold | Live rows selected | Stress total bps | Mean bps |
|---:|---:|---:|---:|
| `0.0` | `8` | `-22.452438548144297` | `-2.806554818518037` |
| `0.5` | `6` | `-14.420744665513263` | `-2.4034574442522105` |
| `1.0` | `5` | `-14.413229620384978` | `-2.8826459240769955` |
| `1.5` | `3` | `-28.04404802215362` | `-9.34801600738454` |
| `2.0` | `2` | `-14.03149345962187` | `-7.015746729810935` |
| `2.5` | `2` | `-14.03149345962187` | `-7.015746729810935` |
| `3.0` | `1` | `-0.014263575631227265` | `-0.014263575631227265` |
| `3.5` | `1` | `-0.014263575631227265` | `-0.014263575631227265` |

Interpretation:

- lowering edge would have created entries;
- but those entries would not have improved the live managed result so far;
- this supports keeping Fast185 unchanged until a new filter is found.

## Decision

Fast186 rejects the shortcut:

> Lowering edge/top-p alone is not a strategy.

The next research direction is not simply “edge lower”. The next useful work is
to find a second filter that distinguishes low-edge correct LONG bursts from
low-edge wrong SHORT bursts without destroying historical daily robustness.

## Artifacts

- Report JSON: `backend/data/microstructure_cross_asset_v1/reports/fast186-fast185-edge-calibration-diagnostic/report.json`
- Report SHA256: `c227a8d60de56e27423b212ea1c9171e3ef6cd16e25f00e182579f681e1cc716`
- Summary SHA256: `7c05ed260195abd16968051c1f5ba8d89482716a837048cef1781abc9580bdb7`
- Historical settlements SHA256: `222cb25f9ebf2965de4d41e791d34a0188d33857880551964ae8c1cb365bafec`
- Live calibration rows SHA256: `aafed9e1c2697f6c72ff5c7e9e639965e48e441be96c394fa3a9990c07b147c5`
