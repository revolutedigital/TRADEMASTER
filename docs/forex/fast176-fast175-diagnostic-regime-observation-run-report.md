# Fast176 — diagnostic regime prospective observation run

Fast176 congelou a melhor pista diagnóstica do Fast175:

`ETHUSDT_qv_30m >= 586337978.3184608`

Ela só pode contar decisões posteriores ao pré-registro:

`2026-09-24T08:53:07-03:00`

Status:

`FAST176_DATA_FRESH_NO_ELIGIBLE_DECISIONS_YET`

## Resposta curta

O observador está pronto e já avaliou a primeira janela pós-âncora, usando
partição parcial diagnóstica coletada via Binance USD-M `fapi`.

Última data comum disponível para BTCUSDT, ETHUSDT e SOLUSDT:

`2026-09-24`

Como a âncora é `2026-09-24T08:53:07-03:00`, o runner avaliou as sete
primeiras decisões fechadas até `2026-09-24T12:25:00Z` e corretamente gerou:

- prospective rows: `7`;
- observações elegíveis: `0`;
- ordens enviadas: `0`.

Isso não aprova nem reprova definitivamente o gate. Só confirma que o trilho
prospectivo está vivo e que a primeira decisão nova não passou os filtros.

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
abs(prediction) >= 17.039926929459345
ETHUSDT_qv_30m >= 586337978.3184608
```

Direção:

- `LONG` se `prediction > 0`;
- `SHORT` se `prediction < 0`.

Custo:

- `2.0` bps por decisão offline.

## Fonte parcial diagnóstica

O arquivo diário Binance Vision de `2026-09-24` ainda não estava disponível
no momento do run. Para não esperar o fechamento do dia, a primeira janela foi
coletada via endpoint público Binance USD-M `fapi/v1/aggTrades`.

Janela:

- início: `2026-09-24T11:20:00+00:00`;
- fim: `2026-09-24T12:40:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `57,535` | `a46146f3483a55759b8379b57acec58c04fb054c40e1b93932ac21da15bb8c20` |
| ETHUSDT | `50,400` | `1b9f74a8367cbf6444aad7e99e7fc8972df5d6adcace5f79be3b546300627953` |
| SOLUSDT | `13,724` | `f250b573051bf1a1e5db9738aa4b10bb5da99a5ce55b6c1c4ff78e20506a1f45` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `7` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T12:25:00+00:00` |
| Prediction | `2.523605614339673` bps |
| Edge threshold | `17.039926929459345` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `115650462.20192` |
| Regime threshold | `586337978.3184608` |
| Regime gate passou? | `false` |
| Elegível? | `false` |
| Actual 15m | `-35.58020638783521` bps |

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

O estado correto agora é:

- a pista de regime foi congelada;
- o threshold de edge foi congelado;
- o runner valida o pré-registro;
- dados históricos não podem mais confirmar essa hipótese;
- as primeiras decisões prospectivas novas foram rejeitadas por edge e regime;
- a sexta decisão teria sido um LONG ruim (`actual 15m = -34.17402698426608` bps), mas ficou fora dos filtros;
- a sétima decisão também teria sido um LONG ruim (`actual 15m = -35.58020638783521` bps), mas ficou fora dos filtros;
- o próximo avanço depende de mais decisões pós-âncora.

Não há estratégia operável ainda.

## Segurança

Fast176 é observação prospectiva offline:

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
  `docs/forex/fast176-fast175-diagnostic-regime-prospective-preregistration.md`;
- runner:
  `backend/scripts/research/run_fast176_fast175_diagnostic_regime_observation.py`;
- teste:
  `backend/tests/unit/test_fast176_fast175_diagnostic_regime_observation.py`;
- coletor parcial diagnóstico:
  `backend/scripts/research/fetch_binance_live_aggtrades_partition.py`;
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/summary.json`;
- diagnóstico prospectivo:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/prospective-diagnostics.json`;
- linhas prospectivas diagnosticadas:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/prospective-diagnostic-rows.jsonl`;
- observações JSONL:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observations.jsonl`;
- observações parquet:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observation-decision-rows.parquet`.

Hashes:

- pré-registro:
  `d464220a86b4fd120a6f91e6ffec69163dcafadf3214c2135105b4a6b9758b92`;
- report:
  `9774502d156f25e2189aa3af8327f1ffe2e0947783201f03c9e69e0aaa48c47c`;
- summary:
  `328756899762cc7ef40fd38ccbec6b3628b59d33bd652b6787a07882d4679c0a`;
- prospective diagnostics:
  `06567dc21a2f060956c737a44b0931965ed8ac5884dc732a7db73a0967147764`;
- prospective diagnostic rows:
  `21940e4fc09c26c0684f3b4fec9b041fb9ceb61165a0da33292ebd7640e16d88`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
