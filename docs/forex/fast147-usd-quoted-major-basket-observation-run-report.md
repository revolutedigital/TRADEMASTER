# Fast147 — USD-quoted major basket observation run

Runner executado depois do pré-registro Fast147.

Status:

`FAST147_OBSERVATION_NO_ELIGIBLE_BASKET_TRADES_YET`

## Resposta curta

A cesta foi pré-registrada e o runner de observação já funciona, mas ainda não
há nenhum trade prospectivo elegível.

O dado M1 local usado pelo runner corrigido termina em:

`2026-09-18T20:59:00+00:00`

Como o pré-registro é:

`2026-09-24T05:49:20-03:00`

o resultado correto é zero observações prospectivas.

## Política observada

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Pares:

- `AUDUSD`;
- `EURUSD`;
- `GBPUSD`;
- `NZDUSD`.

Regra:

- timeframe de sinal: `D1`;
- execução offline: M1;
- entrada: `d1_donchian_fade_ch100_hold20`;
- risco: `stop2_fixed_1r`;
- lado: somente `SHORT`;
- gestão: `fixed_1r`;
- custo primário de decisão: `STRESS`.

## Resultado do runner

| Métrica | Resultado |
|---|---:|
| Raw M1 start | `2019-01-01T22:00:00+00:00` |
| Raw M1 end | `2026-09-18T20:59:00+00:00` |
| Raw M1 rows | `11206603` |
| Sinais reconstruídos | `344` |
| Trades materializados | `281` |
| Observações pós-pré-registro | `0` |
| Ordens enviadas | `0` |

Sinais por par:

| Par | Sinais |
|---|---:|
| `AUDUSD` | `90` |
| `EURUSD` | `85` |
| `GBPUSD` | `90` |
| `NZDUSD` | `79` |

Trades materializados por par:

| Par | Trades |
|---|---:|
| `AUDUSD` | `71` |
| `EURUSD` | `73` |
| `GBPUSD` | `72` |
| `NZDUSD` | `65` |

## Gates

Como `eligible_observation_count=0`, nenhum gate de promoção pode passar.

Status reasons:

- `not_approved_strategy`;
- `research_only`;
- `no_order_execution`;
- `basket_observation`;
- `no_post_preregistration_basket_trades`.

Shadow gate pendente por:

- `insufficient_stress_trades`;
- `stress_mean_below_threshold`;
- `stress_lower_confidence_bound_not_positive`;
- `base_mean_not_positive`;
- `insufficient_prospective_years`;
- `positive_year_share_below_threshold`;
- `insufficient_nonnegative_pairs`.

## Segurança

Fast147 observation é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast147_usd_quoted_major_basket_observation.py`;
- teste:
  `backend/tests/unit/test_fast147_usd_quoted_major_basket_observation.py`;
- report JSON:
  `backend/data/lab_fast147/usd-quoted-major-basket-observation/report.json`;
- sinais:
  `backend/data/lab_fast147/usd-quoted-major-basket-observation/entry_signals.parquet`;
- trades materializados:
  `backend/data/lab_fast147/usd-quoted-major-basket-observation/materialized_trade_rows.parquet`;
- observações:
  `backend/data/lab_fast147/usd-quoted-major-basket-observation/observations.jsonl`.

Hashes:

- report:
  `2efb19322e7301ba24915dfafdb2f9384d571a74c659c7ab3c86d1ac2cd2222a`;
- sinais:
  `1d3d8bb7fa92a7c4f462d1014b2662dc475a594845079527ee1b936aa6e0991c`;
- trades materializados:
  `b22e95fb9d3baf9f8d4b6612ac3637249d6c831cd908eb0c4d83d1b39acb37da`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Nota de correção: o runner inicial usava o loader histórico que cortava a base
em `2026-09-01`. O runner atual carrega M1 de observação sem esse corte, no
mesmo padrão dos observadores Fast134/Fast143. Com isso, setembro parcial passa
a ser considerado corretamente.

## Próxima ação

Quando houver M1 posterior a `2026-09-24T05:49:20-03:00` para os quatro pares,
rerodar Fast147. Até lá, a candidata está congelada, mas sem evidência
prospectiva.
