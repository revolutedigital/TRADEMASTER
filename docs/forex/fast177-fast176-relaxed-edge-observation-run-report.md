# Fast177 — relaxed Fast176 prospective observation run

Fast177 congelou uma versão menos conservadora do Fast176 para testar,
somente em dados futuros, se o gate anterior estava perdendo movimento demais.

Âncora prospectiva:

`2026-09-24T10:05:09.742000-03:00`

Status:

`FAST177_WAITING_FOR_POST_PREREGISTRATION_DATA`

## Resposta curta

O experimento está pré-registrado, tem registry próprio e o runner validou o
hash do pré-registro. Com a partição local atual até
`2026-09-24T13:05:59.999000+00:00`, ainda não existe decisão pós-âncora com
outcome completo de `900s`.

Isso é esperado: a primeira decisão estritamente posterior à âncora é
`2026-09-24T13:10:00Z`, e ela só pode ser avaliada depois da janela futura
fechar em torno de `2026-09-24T13:25:00Z`.

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
- fim: `2026-09-24T13:05:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `77,917` | `aa2ebbf1ee8c82f43b04b0f496206d8cbaecb37fa5a1067392d7f33c6b8720b7` |
| ETHUSDT | `67,328` | `cbc8aa5f26450ca45e37fe3668f2c0a83fd909915aa54c6ae6b58cded92befe0` |
| SOLUSDT | `18,087` | `6de78a6c440b39a11da159b237a87a1313957a58e2b3b81bf37a20ede3d23e3e` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `0` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

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
A decisão de `2026-09-24T12:50:00Z`, por exemplo, teria passado nos thresholds
relaxados (`prediction = 8.930841707010687`, `ETHUSDT_qv_30m = 127105151.37745`)
e teve `actual 15m = 44.084062702385296` bps. Ela não conta para Fast177 porque
aconteceu antes da âncora do Fast177.

O teste honesto começa na próxima linha com outcome completo após
`2026-09-24T13:25:00Z`.

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
  `98231bf7f35bf71435f992cd68c05387591ee482f761ac2011d8ba7c0e2aaeb4`;
- summary:
  `fa52a02e2a845bcd072362f35f22cdd3b803180de2a9c8cc4b91a9376f51408b`;
- prospective diagnostics:
  `7f6b6966fce13433d12f6de5c59629dc79a7b79b9d30d88c77abc0cdb5c96c82`;
- prospective diagnostic rows:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
