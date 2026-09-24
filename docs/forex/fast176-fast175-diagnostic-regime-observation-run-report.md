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

Como a âncora é `2026-09-24T08:53:07-03:00`, o runner avaliou as dezesseis
primeiras decisões fechadas até `2026-09-24T13:10:00Z` e corretamente gerou:

- prospective rows: `16`;
- observações elegíveis: `0`;
- ordens enviadas: `0`.

Isso não aprova nem reprova definitivamente o gate. Só confirma que o trilho
prospectivo está vivo e que as decisões novas ainda não passaram os filtros.

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
| Prospective rows pós-âncora | `16` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

Diagnóstico da decisão pós-âncora:

| Métrica | Valor |
|---|---:|
| Decision time | `2026-09-24T13:10:00+00:00` |
| Prediction | `5.6138486736465945` bps |
| Edge threshold | `17.039926929459345` bps |
| Edge gate passou? | `false` |
| ETHUSDT_qv_30m | `127251329.3691` |
| Regime threshold | `586337978.3184608` |
| Regime gate passou? | `false` |
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
- a décima segunda decisão reforçou o diagnóstico: `prediction = 8.930841707010687` bps e `actual 15m = 44.084062702385296` bps, mas ainda ficou abaixo do edge congelado de Fast176 e do regime rígido;
- as decisões seguintes continuaram positivas, mas sem edge suficiente para o Fast176 rígido; a mais recente foi `prediction = 5.6138486736465945` bps e `actual 15m = 27.013382361806837` bps;
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
  `0e72c82c7dd198272b9ea9d0e4404b22d495bf82a236bfee9b0cefee48a2df8d`;
- summary:
  `ea509b53214f27c0652fb38fe69599a481c16e804fc14b1e13070ddbab195710`;
- prospective diagnostics:
  `2a446ca6650696b5f9f471c736cf39ffb4905995bba0bceafc1fa3f0348df404`;
- prospective diagnostic rows:
  `2c89658d346682cd8269b7e5b0c250f113e33df27c845ff4c1bfffa2f42b535d`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
