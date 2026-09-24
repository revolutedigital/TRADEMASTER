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
`2026-09-24T13:35:59.999000+00:00`, existem 3 decisões pós-âncora com outcome
completo de `900s`.

A terceira decisão passou no regime e no edge relaxados, gerando a primeira
entrada elegível do Fast177:

- decision time: `2026-09-24T13:20:00+00:00`;
- prediction: `7.876874755717758` bps;
- edge threshold: `7.5` bps;
- `ETHUSDT_qv_30m`: `232377412.03993`;
- actual 15m: `0.8631300849626378` bps;
- net após custo: `-1.1368699150373622` bps;
- elegível: `true`.

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
- fim: `2026-09-24T13:35:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `131,501` | `d5bc0921770c3b5877a4a0fc5aaf837e1aec3f73f8624a863dc88678ac022bb1` |
| ETHUSDT | `113,260` | `181f16654d702b4da85ee4596b6cf9373f747502f9200acbbcb3f23c62d88873` |
| SOLUSDT | `27,187` | `ee467f0eaeb9e3f71a5b4183d676f558931183a9e8c130f0f01c01cef1e7f7c8` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `3` |
| Observações elegíveis | `1` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T13:20:00+00:00` |
| Prediction | `7.876874755717758` bps |
| Edge threshold | `7.5` bps |
| Edge gate passou? | `true` |
| ETHUSDT_qv_30m | `232377412.03993` |
| Regime threshold | `100000000.0` |
| Regime gate passou? | `true` |
| Elegível? | `true` |
| Actual 15m | `0.8631300849626378` bps |
| Net pós-custo | `-1.1368699150373622` bps |

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
bps). Isso não encerra Fast177, mas mostra que o `prediction` ainda não está
bem calibrado como ranking de payoff líquido.

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
  `626cc14e20a6a90b2e29a5b83c7e2ae885c9beed325dd084ed691e68c76ac887`;
- summary:
  `5c82eb4de9d1bccafa32ab77c5a13d46bd85ed86627289faffe9196d68cd17d6`;
- prospective diagnostics:
  `68d8156b745c765d1bf27ed3a4669f98f3cb7cd5f8f9bc0fec15b015d49e3174`;
- prospective diagnostic rows:
  `2e77951fe4263b3820b754101eb56628be311e3838dfec5a85ff531a6b57f469`;
- observations:
  `2cf5c6b497c4ff9be49b1a49736f753f70a20cf321b81fad13c3f4f29d144e05`;
- observation decision rows:
  `ee276e8e4899cb99dcd31e0b51ebba2e2624faf93536bbf706c53b8ba41a6ea8`.
