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

Como a âncora é `2026-09-24T08:53:07-03:00`, o runner avaliou as três
primeiras decisões fechadas até `2026-09-24T12:05:00Z` e corretamente gerou:

- prospective rows: `3`;
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
- fim: `2026-09-24T12:20:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `39,984` | `19490d4b0214707fd8e59db2935308c772ec8b2037d3c453af91f04750f3903f` |
| ETHUSDT | `34,325` | `46e33c9a0fafea53f1898ec657523cb40156fa9201db103756bb21c41f4e2ed0` |
| SOLUSDT | `10,557` | `dab1800fe1878ad49d180a10ee659f8bd82173c2e0e5ceb5ed4f7c1f4f4b3ee1` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `3` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T12:05:00+00:00` |
| Prediction | `1.3775142838215286` bps |
| Edge threshold | `17.039926929459345` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `119521560.81124` |
| Regime threshold | `586337978.3184608` |
| Regime gate passou? | `false` |
| Elegível? | `false` |

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
- a primeira decisão prospectiva nova foi rejeitada por edge e regime;
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
- observações JSONL:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observations.jsonl`;
- observações parquet:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observation-decision-rows.parquet`.

Hashes:

- pré-registro:
  `d464220a86b4fd120a6f91e6ffec69163dcafadf3214c2135105b4a6b9758b92`;
- report:
  `afe65f750d7393496f19d4d5d2b97608069bdf77a57e6358056ab56e9d86daf2`;
- summary:
  `b203817a8660c1580ee63a0c74e3ee1f8d6468714cf56bbf356f62a04bc2d90d`;
- prospective diagnostics:
  `63c56a9bb38712c41300b8a8e4ac5798b87b61c509a6e91fb20ba1e5ad52bc3b`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
