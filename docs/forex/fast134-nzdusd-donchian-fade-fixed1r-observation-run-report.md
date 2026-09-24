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
- fim do M1 local usado: `2026-09-18T20:59:00+00:00`;
- timestamp de pré-registro: `2026-09-24T04:04:40-03:00`.

Não há trades elegíveis porque, mesmo após importar o mês `2026-09` de
NZDUSD, o dado M1 local ainda termina antes do pré-registro. Isso não reprova
nem aprova a hipótese; só confirma que ainda não existe evidência prospectiva
nova para essa candidata.

Observação de dado: `NZDUSD_M1_202609_202609.parquet` tem `19.610` minutos de
`2026-09-01T04:00:00+00:00` a `2026-09-18T20:59:00+00:00`. O H1 oracle local
termina em agosto, então a validação mensal de setembro fica marcada como
unverified/fail-closed, não como aprovação de qualidade.

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
  `8795cce08d4e7e456708a490fd9efab2faf4922deef0432080dae8960e1552b3`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades materializados:
  `b1dab2d1aba799b4545cba4719e4c407eb9fb1a01a7d08fa771c46035d3214f0`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Interpretação

O estado correto continua sendo:

- temos uma candidata estatística pré-registrada;
- ainda não temos uma estratégia aprovada;
- ainda não há Testnet, Demo nem LIVE;
- o próximo avanço depende de importar/capturar M1 de NZDUSD realmente posterior a
  `2026-09-24T04:04:40-03:00` e rerodar o Fast134.
