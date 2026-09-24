# Fast177 — relaxed Fast176 prospective observation run

Fast177 congelou uma versão menos conservadora do Fast176 para testar,
somente em dados futuros, se o gate anterior estava perdendo movimento demais.

Âncora prospectiva:

`2026-09-24T10:05:09.742000-03:00`

Status:

`FAST177_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET`

## Resposta curta

O experimento está pré-registrado, tem registry próprio e o runner validou o
hash do pré-registro. Com a partição local atual até
`2026-09-24T13:25:59.999000+00:00`, existe 1 decisão pós-âncora com outcome
completo de `900s`.

Essa primeira decisão passou no regime relaxado, mas não passou no edge
relaxado:

- decision time: `2026-09-24T13:10:00+00:00`;
- prediction: `5.613848673646596` bps;
- edge threshold: `7.5` bps;
- `ETHUSDT_qv_30m`: `127251329.3691`;
- actual 15m: `27.013382361806837` bps;
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
- fim: `2026-09-24T13:25:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `100,094` | `a997e203ec2350bb7543c5cb905973d2db0312d83f23b40a961aed53b474bec9` |
| ETHUSDT | `88,326` | `7df477cb41a7e980fe0c9214f1a4cd75284547278740d8b22583b04aafad269e` |
| SOLUSDT | `23,084` | `5b1d8a0c86a54cdc67ee999d093f05ae6aa569c30d837e6e48fb71a4969366e3` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `1` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T13:10:00+00:00` |
| Prediction | `5.613848673646596` bps |
| Edge threshold | `7.5` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `127251329.3691` |
| Regime threshold | `100000000.0` |
| Regime gate passou? | `true` |
| Elegível? | `false` |
| Actual 15m | `27.013382361806837` bps |

Resumo de observações:

| Métrica | Valor |
|---|---:|
| decision_count | `0` |
| day_count | `0` |
| net_total_bps | `0.0` |
| net_mean_bps | `0.0` |
| net_lcb_bps | `0.0` |
| positive_day_share | `0.0` |

## Interpretação

Fast177 existe porque as últimas decisões do Fast176 apontaram um problema
real: o gate rígido evitou entradas ruins, mas também perdeu movimentos grandes.
As linhas pré-âncora que teriam passado no gate relaxado somavam
`+58.26957986522819` bps líquidos, mas não contam como validação Fast177.

O primeiro teste honesto pós-âncora também perdeu um movimento positivo:
`actual 15m = 27.013382361806837` bps. O regime relaxado funcionou, mas o edge
de `7.5` bps ainda barrou a entrada (`prediction = 5.613848673646596` bps).
Isso não reprova Fast177, mas indica que o problema pode estar no cutoff de
edge, não apenas no regime.

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
  `ca6dfa9a9fc477b6c5a88b34b8947f3ab1909d9a426476fdbfacfa46df214e73`;
- summary:
  `7024038f6eea24691e5725167e16eb984eddbab59d2673d4b8fb2e17bf7e4506`;
- prospective diagnostics:
  `8e0d5b14b84920be5ef989a56e036cc38d4e687c9f8db932c929cb38592a7223`;
- prospective diagnostic rows:
  `b38a31de5828832b42bc04b8a5f34f7593a5629d375e44599cbd1ce44e3620a7`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
