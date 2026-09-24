# Fast143 — NZDUSD Donchian fade short-only observation run

Fast143 implementou e rodou o observador prospectivo offline da candidata:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Status:

`FAST143_OBSERVATION_NO_ELIGIBLE_SHORT_TRADES_YET`

## Regra observada

- par: `NZDUSD`;
- timeframe de sinal: `D1`;
- sinal: Donchian fade;
- canal: `100` dias;
- lado permitido: somente `SHORT`;
- condição elegível: fechamento D1 acima da máxima causal dos `100` dias
  anteriores;
- entrada: primeiro M1 disponível depois do fechamento D1;
- stop: `2.0 * ATR(14)` diário causal;
- alvo: `1R`;
- hold máximo: `20` dias;
- cenário primário de decisão: `STRESS`.

## Resultado da execução inicial

- status: `FAST143_OBSERVATION_NO_ELIGIBLE_SHORT_TRADES_YET`;
- M1 local usado:
  - início: `2019-01-01T22:02:00+00:00`;
  - fim: `2026-09-18T20:59:00+00:00`;
  - linhas: `2,811,188`;
- sinais D1 reconstruídos: `79`;
- trades históricos materializados pela regra base: `65`;
- observações prospectivas `SHORT` elegíveis: `0`;
- ordens enviadas: `0`.

Não há trade elegível porque o M1 local ainda termina antes do pré-registro
Fast143 (`2026-09-24T05:25:14-03:00`). Isso não aprova nem reprova a
hipótese; só confirma que o observador está pronto e fail-closed.

## Gates

Gate de falha precoce:

- `early_fail=false`;
- nenhum trade prospectivo, então não há perda prospectiva a avaliar.

Gate para shadow offline:

- `shadow_candidate=false`;
- motivos:
  - `insufficient_stress_trades`;
  - `stress_mean_below_threshold`;
  - `stress_lower_confidence_bound_not_positive`;
  - `base_mean_not_positive`;
  - `insufficient_prospective_years`;
  - `positive_year_share_below_threshold`.

Esses motivos são esperados com `0` observações.

## Validação do dado

O runner leu três arquivos NZDUSD:

| Arquivo | Hash |
|---|---|
| `NZDUSD_M1_201901_202108.parquet` | `7b82a35af74cd284579b3b8795f5b5c0de2dd0d004b531e5a22c32081b5e6ae5` |
| `NZDUSD_M1_202109_202608.parquet` | `3222c583e7fe51fefab1d712f5c2ad79d6a87629ec0f143299c4b9a589755172` |
| `NZDUSD_M1_202609_202609.parquet` | `e9fdae6fe9b209fe01243de303bd25a02bcf9c67980348f3ae19fb425754ee72` |

O CSV de validação atual de setembro existe, mas não traz coluna `passes`;
portanto o runner trata como `unverified/fail-closed`, não como aprovação de
qualidade:

`validation_csv_without_passes_column_treated_as_unverified`

## Segurança

Fast143 permanece estritamente offline:

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

- pré-registro:
  `docs/forex/fast143-nzdusd-donchian-fade-short-only-prospective-preregistration.md`;
- runner:
  `backend/scripts/research/run_fast143_nzdusd_donchian_fade_short_only_observation.py`;
- teste:
  `backend/tests/unit/test_fast143_nzdusd_donchian_fade_short_only_observation.py`;
- relatório JSON:
  `backend/data/lab_fast143/nzdusd-donchian-fade-short-only-observation/report.json`;
- sinais:
  `backend/data/lab_fast143/nzdusd-donchian-fade-short-only-observation/entry_signals.parquet`;
- trades materializados:
  `backend/data/lab_fast143/nzdusd-donchian-fade-short-only-observation/materialized_trade_rows.parquet`;
- observações:
  `backend/data/lab_fast143/nzdusd-donchian-fade-short-only-observation/observations.jsonl`.

Hashes:

- pré-registro:
  `4191ff3601d085737f0e04f55b77348c754b0866a1d64ec5a411e928f7a0c253`;
- report:
  `87b632982a9f63174d8be0770e3a19bbdf39150cdd30a9624982065e03b81023`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades materializados:
  `b1dab2d1aba799b4545cba4719e4c407eb9fb1a01a7d08fa771c46035d3214f0`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O sistema agora está preparado para observar a candidata `short_only` quando
houver M1 de NZDUSD posterior a `2026-09-24T05:25:14-03:00`. Até lá, o status
correto é aguardar dado prospectivo novo.
