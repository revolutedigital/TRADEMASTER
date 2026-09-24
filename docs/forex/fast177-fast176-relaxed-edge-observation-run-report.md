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
`2026-09-24T13:30:59.999000+00:00`, existem 2 decisões pós-âncora com outcome
completo de `900s`.

As duas decisões passaram no regime relaxado, mas nenhuma passou no edge
relaxado. A mais recente:

- decision time: `2026-09-24T13:15:00+00:00`;
- prediction: `2.0467815583232944` bps;
- edge threshold: `7.5` bps;
- `ETHUSDT_qv_30m`: `135650920.51112`;
- actual 15m: `22.966953941845336` bps;
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
- fim: `2026-09-24T13:30:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `106,518` | `7d41b65d0a6c547bd73d07c85057d4d7714450439b705d180abc82ae4836189d` |
| ETHUSDT | `92,904` | `10ae9b86e8d2df3696ee132c251f1b3d1b041c76328427865105466988a850ce` |
| SOLUSDT | `24,246` | `489cb54b4925fb1bd690abc2d9a6f71def504d4a1e317c3ccb8946512e76177d` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `2` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T13:15:00+00:00` |
| Prediction | `2.0467815583232944` bps |
| Edge threshold | `7.5` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `135650920.51112` |
| Regime threshold | `100000000.0` |
| Regime gate passou? | `true` |
| Elegível? | `false` |
| Actual 15m | `22.966953941845336` bps |

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

Os dois primeiros testes honestos pós-âncora também perderam movimentos
positivos: `27.013382361806837` bps e `22.966953941845336` bps. O regime
relaxado funcionou nas duas linhas, mas o edge de `7.5` bps barrou ambas.
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
  `49e94b459a9b410cad3279c55498006bc396e24982c35ecec8d281833537e160`;
- summary:
  `5e7e09fc0ac9080d2fd2f849696971288dc54154902e9a4806232160bd0a4a04`;
- prospective diagnostics:
  `e92096fcf10bb972803b7cbe8e9b909ea931099c690c3e84d83bed01d3be4f73`;
- prospective diagnostic rows:
  `d3b430e41de05c7aadf16c61f66663f9e1a6c57e4edbe61404c0fe32fd9f3fca`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
