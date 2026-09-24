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

Como a âncora é `2026-09-24T08:53:07-03:00`, o runner avaliou as onze
primeiras decisões fechadas até `2026-09-24T12:45:00Z` e corretamente gerou:

- prospective rows: `11`;
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
- fim: `2026-09-24T13:00:59.999000+00:00`;
- qualidade: `DIAGNOSTIC_LIVE_PARTIAL_VALID`;
- gaps de sequência: `0`;
- duplicatas: `0`.

Partições normalizadas:

| Símbolo | Rows | Normalized SHA256 |
|---|---:|---|
| BTCUSDT | `71,153` | `e723df567b6d932a8575b4499d009d14b2bd8f843af73a7130c384c62e57ae04` |
| ETHUSDT | `62,111` | `cecebe82c6adb64cf47e8f31b92525c6e77421f488ef8d33ef6d2f5654eba75f` |
| SOLUSDT | `16,538` | `d8cf3340fe7e3609fb45737d90959fff3bd62a17ee53142daaef7b0d04bcf776` |

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `11` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T12:45:00+00:00` |
| Prediction | `4.99745912186051` bps |
| Edge threshold | `17.039926929459345` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `130278185.12409` |
| Regime threshold | `586337978.3184608` |
| Regime gate passou? | `false` |
| Elegível? | `false` |
| Actual 15m | `28.805353783494247` bps |

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
- a oitava decisão também teria sido um LONG ruim (`actual 15m = -19.305019305019265` bps), mas ficou fora dos filtros;
- a nona decisão foi o maior edge prospectivo até agora (`7.9990196390228006` bps), mas ainda ficou abaixo do threshold congelado e com regime baixo;
- a décima decisão foi um miss conservador: `actual 15m = 15.449161655787247` bps, mas edge e regime ainda ficaram abaixo dos gates congelados;
- a décima primeira decisão foi outro miss conservador forte: `actual 15m = 28.805353783494247` bps, mas edge e regime ainda ficaram abaixo dos gates congelados;
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
  `f455e9182ef7e982dba2da0ce4daf82e7e87a8a0db1147d6f7c914c429888f05`;
- summary:
  `6bc3f9e710696cdc8284f879a4a35a9a56e03b917637638b97278b6cd04e04e4`;
- prospective diagnostics:
  `7d2bcefb43b314698fcbc42e5a2e4320924b0f2a6120fa1afc69cac429857f3f`;
- prospective diagnostic rows:
  `729e2a97b6acef36187294b29e480cce80b8482bf99ddf8c3491cb3bd7b0e00f`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
