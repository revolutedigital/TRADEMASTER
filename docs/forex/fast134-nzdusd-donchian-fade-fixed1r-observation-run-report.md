# Fast134 — NZDUSD Donchian fade fixed 1R observation run

Fast134 executou o runner prospectivo offline da candidata congelada em
`docs/forex/fast134-nzdusd-donchian-fade-fixed1r-prospective-preregistration.md`.

Status:

`FAST134_OBSERVATION_NO_ELIGIBLE_TRADES_YET`

## Regra observada

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

- par: `NZDUSD`;
- timeframe de sinal: `D1`;
- sinal: fade/contra rompimento Donchian de `100` dias;
- entrada: primeiro M1 disponível depois do fechamento D1;
- stop: `2.0 * ATR(14)` diário causal;
- alvo: `1R`;
- hold máximo: `20` dias;
- cenário primário de decisão: `STRESS`.

## Resultado da execução

- sinais reconstruídos: `79`;
- trades históricos materializados pela regra congelada: `65`;
- trades elegíveis pós-pré-registro: `0`;
- início do M1 local usado: `2019-01-01T22:02:00+00:00`;
- fim do M1 local usado: `2026-08-31T23:59:00+00:00`;
- timestamp de pré-registro: `2026-09-24T04:04:40-03:00`.

Não há trades elegíveis porque o dado M1 local termina antes do pré-registro.
Isso não reprova nem aprova a hipótese; só confirma que ainda não existe
evidência prospectiva nova para essa candidata.

## Gates

- early fail: `false`;
- shadow candidate: `false`;
- motivo: ausência de trades prospectivos;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

## Artefatos

- runner:
  `backend/scripts/research/run_fast134_nzdusd_donchian_fade_fixed1r_observation.py`;
- teste:
  `backend/tests/unit/test_fast134_nzdusd_donchian_fade_fixed1r_observation.py`;
- report JSON:
  `backend/data/lab_fast134/nzdusd-donchian-fade-fixed1r-observation/report.json`;
- sinais:
  `backend/data/lab_fast134/nzdusd-donchian-fade-fixed1r-observation/entry_signals.parquet`;
- trades materializados:
  `backend/data/lab_fast134/nzdusd-donchian-fade-fixed1r-observation/materialized_trade_rows.parquet`;
- observações:
  `backend/data/lab_fast134/nzdusd-donchian-fade-fixed1r-observation/observations.jsonl`.

Hashes:

- report:
  `f00eb574250d8098b64b8d7707a6e8135c33f88d40296b683aab2c894c9618f8`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades materializados:
  `84eeb1903e4faaf77458842260e8301ca71c80b55943a5b64a5c87ee410654f4`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Interpretação

O estado correto continua sendo:

- temos uma candidata estatística pré-registrada;
- ainda não temos uma estratégia aprovada;
- ainda não há Testnet, Demo nem LIVE;
- o próximo avanço depende de importar/capturar M1 de NZDUSD posterior a
  `2026-09-24T04:04:40-03:00` e rerodar o Fast134.
