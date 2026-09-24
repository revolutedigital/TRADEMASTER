# Fast147 — USD-quoted major basket observation run

Runner executado depois do pré-registro Fast147.

Status:

`FAST147_OBSERVATION_NO_ELIGIBLE_BASKET_TRADES_YET`

## Resposta curta

A cesta foi pré-registrada e o runner de observação já funciona, mas ainda não
há nenhum trade prospectivo elegível.

O dado M1 local usado pelo runner corrigido, depois do refresh local de setembro
para os quatro pares da cesta, termina em:

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
| Raw M1 min pair end | `2026-09-18T20:59:00+00:00` |
| Raw M1 rows | `11265417` |
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

Fim do M1 por par:

| Par | Último minuto |
|---|---|
| `AUDUSD` | `2026-09-18T20:59:00+00:00` |
| `EURUSD` | `2026-09-18T20:59:00+00:00` |
| `GBPUSD` | `2026-09-18T20:59:00+00:00` |
| `NZDUSD` | `2026-09-18T20:59:00+00:00` |

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
  `a5f2a488da89ece0bd9f9a46c4fad67d7a82be9b18d79e0e08292a28214067ec`;
- sinais:
  `1d3d8bb7fa92a7c4f462d1014b2662dc475a594845079527ee1b936aa6e0991c`;
- trades materializados:
  `2f9146cc6435e393c83d6235dd7ac0dfc6b2fe386b0ce9984d7838c6e9c394b0`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Nota de correção: o runner inicial usava o loader histórico que cortava a base
em `2026-09-01`. O runner atual carrega M1 de observação sem esse corte, no
mesmo padrão dos observadores Fast134/Fast143. Com isso, setembro parcial passa
a ser considerado corretamente.

Hashes dos parquets locais de setembro usados no refresh:

- `AUDUSD_M1_202609_202609.parquet`:
  `9f3f4aa352fae44f39bd637ef56ad4cf6ff29b0546886bea94112521bc6d3b36`;
- `EURUSD_M1_202609_202609.parquet`:
  `83e10b41c3b2d69281d732b0f468a8bbcb53750f7acf158e1f02a6f15448564b`;
- `GBPUSD_M1_202609_202609.parquet`:
  `c98fa9becf27f5cc29287bb7b51b68f8377fadc4c12880496669a4161bce9266`;
- `NZDUSD_M1_202609_202609.parquet`:
  `e9fdae6fe9b209fe01243de303bd25a02bcf9c67980348f3ae19fb425754ee72`.

## Próxima ação

Quando houver M1 posterior a `2026-09-24T05:49:20-03:00` para os quatro pares,
rerodar Fast147. Até lá, a candidata está congelada, mas sem evidência
prospectiva.
