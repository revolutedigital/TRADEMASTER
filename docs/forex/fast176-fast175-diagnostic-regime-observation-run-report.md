# Fast176 — diagnostic regime prospective observation run

Fast176 congelou a melhor pista diagnóstica do Fast175:

`ETHUSDT_qv_30m >= 586337978.3184608`

Ela só pode contar decisões posteriores ao pré-registro:

`2026-09-24T08:53:07-03:00`

Status:

`FAST176_WAITING_FOR_POST_PREREGISTRATION_DATA`

## Resposta curta

O observador está pronto, mas ainda não há dado local posterior à âncora.

Última data comum disponível para BTCUSDT, ETHUSDT e SOLUSDT:

`2026-09-22`

Como a âncora é `2026-09-24T08:53:07-03:00`, o runner corretamente gerou:

- prospective rows: `0`;
- observações elegíveis: `0`;
- ordens enviadas: `0`.

Isso não aprova nem reprova o gate. Só confirma que agora existe um trilho
fail-closed para medir a hipótese em dados novos.

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

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-22` |
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

O estado correto agora é:

- a pista de regime foi congelada;
- o threshold de edge foi congelado;
- o runner valida o pré-registro;
- dados históricos não podem mais confirmar essa hipótese;
- o próximo avanço depende de partições públicas pós-âncora.

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
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/summary.json`;
- observações JSONL:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observations.jsonl`;
- observações parquet:
  `backend/data/microstructure_cross_asset_v1/reports/fast176-fast175-diagnostic-regime-observation/observation-decision-rows.parquet`.

Hashes:

- pré-registro:
  `d464220a86b4fd120a6f91e6ffec69163dcafadf3214c2135105b4a6b9758b92`;
- report:
  `6a2e0cbab28fb1d4f94ba5d370cdef07b75e8c16c0c1fb039a5e9f21ab0272af`;
- summary:
  `a50f4f3647713b79b52cd4d19706e862b02479b5c9273216bcf613e2340c8070`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `34216d645f3e033067dd3f069743c72df402702827b2c524bc975ce60292b102`.
