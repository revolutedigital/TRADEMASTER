# Fast177 — relaxed Fast176 prospective observation run

Fast177 congelou uma versão menos conservadora do Fast176 para testar,
somente em dados futuros, se o gate anterior estava perdendo movimento demais.

Âncora prospectiva:

`2026-09-24T10:05:09.742000-03:00`

Status:

`FAST177_OBSERVATION_INSUFFICIENT_PROSPECTIVE_EVIDENCE`

## Resposta curta

O experimento está pré-registrado, tem registry próprio e o runner validou o
hash do pré-registro. Com a partição local atual até
`2026-09-24T13:40:59.999000+00:00`, existem 4 decisões pós-âncora com outcome
completo de `900s`.

A quarta decisão passou no regime, mas não passou no edge. Ela teria sido SHORT
contra alta de ETH, então o cutoff evitou uma perda:

- decision time: `2026-09-24T13:25:00+00:00`;
- prediction: `-1.4963706318144414` bps;
- edge threshold: `7.5` bps;
- `ETHUSDT_qv_30m`: `252131083.19257`;
- actual 15m: `17.021052750234578` bps;
- elegível: `false`.

## Política congelada

Target:

- `ETHUSDT`.

Modelo:

- `Ridge(alpha=10.0)`;
- `SimpleImputer(strategy="constant", fill_value=0.0)`;
- `StandardScaler`;
- treino somente em `fast50`, `fast51`, `fast52`.

Entrada:

```text
abs(prediction) >= 7.5
ETHUSDT_qv_30m >= 100000000.0
```

Direção:

- `LONG` se `prediction > 0`;
- `SHORT` se `prediction < 0`.

Custo:

- `2.0` bps por decisão offline.

## Fonte parcial diagnóstica

Janela:

- início: `2026-09-24T11:20:00+00:00`;
- fim: `2026-09-24T13:40:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `148,525` | `b3dc3ac4f394ceaba3007c2ea712bea52c252c7ad9f3415c73f221bf7f6be22d` |
| ETHUSDT | `124,858` | `5d970243c89565826c093afbedb078d0ee5bb434de98199285a28a2809bc0151` |
| SOLUSDT | `28,902` | `a980116c202b744d3e21a4b2a630e3c43e621ef08bba663d88738c7a31d3d2d1` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `4` |
| Observações elegíveis | `1` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T13:25:00+00:00` |
| Prediction | `-1.4963706318144414` bps |
| Edge threshold | `7.5` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `252131083.19257` |
| Regime threshold | `100000000.0` |
| Regime gate passou? | `true` |
| Elegível? | `false` |
| Actual 15m | `17.021052750234578` bps |

Resumo de observações:

| Métrica | Valor |
|---|---:|
| decision_count | `1` |
| day_count | `1` |
| net_total_bps | `-1.1368699150373622` |
| net_mean_bps | `-1.1368699150373622` |
| net_lcb_bps | `-1.1368699150373622` |
| positive_day_share | `0.0` |

## Interpretação

Fast177 existe porque as últimas decisões do Fast176 apontaram um problema
real: o gate rígido evitou entradas ruins, mas também perdeu movimentos grandes.
As linhas pré-âncora que teriam passado no gate relaxado somavam
`+58.26957986522819` bps líquidos, mas não contam como validação Fast177.

Os dois primeiros testes honestos pós-âncora perderam movimentos positivos
porque o edge ficou abaixo de `7.5` bps. A terceira decisão finalmente entrou,
mas o movimento foi pequeno demais para pagar custo (`net = -1.1368699150373622`
bps). A quarta decisão teria sido um SHORT ruim e ficou fora por edge baixo.
Isso não encerra Fast177, mas mostra que o `prediction` ainda não está bem
calibrado como ranking de payoff líquido.

Não há estratégia operável ainda.

## Segurança

Fast177 é observação prospectiva offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- pré-registro:
  `docs/forex/fast177-fast176-relaxed-edge-prospective-preregistration.md`;
- registry:
  `docs/forex/fast177-registry.jsonl`;
- runner:
  `backend/scripts/research/run_fast177_fast176_relaxed_edge_observation.py`;
- teste:
  `backend/tests/unit/test_fast177_fast176_relaxed_edge_observation.py`;
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/summary.json`;
- diagnóstico prospectivo:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/prospective-diagnostics.json`;
- linhas prospectivas diagnosticadas:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/prospective-diagnostic-rows.jsonl`;
- observações JSONL:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/observations.jsonl`;
- observações parquet:
  `backend/data/microstructure_cross_asset_v1/reports/fast177-fast176-relaxed-edge-observation/observation-decision-rows.parquet`.

Hashes:

- pré-registro:
  `72f78d9b9eda9d3f743a8027fa81460476d4bd8d2f8313a7e2f1f19aa9bd7c47`;
- report:
  `ae3eb23bfac5f07823bfad82638950d5c4bfd8678f9ae9403b70dc42f7374c1b`;
- summary:
  `abfcaaaa200352fa4b4a92eac9824ea51277c32d507341272ed8a70d261cba23`;
- prospective diagnostics:
  `16063fa20e309ff302410a64ab5f04a53ef4206e744975b4c7ae1dec79454619`;
- prospective diagnostic rows:
  `f221fc2781a5dbf4a5d69e28d3728dcf1851adf10fa8aa9bda14a33c4c50aaff`;
- observations:
  `99163ad399b920131e97d05fc12969d05e38e2f7685156901bf4a7c354e15c87`;
- observation decision rows:
  `d54669b8d787d8a92c81bec6dc87a17a9bf66aeae52a1ac1643e8bf67628a288`.
