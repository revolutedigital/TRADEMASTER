# Fast131 — NZDUSD Donchian fade observation run

Fast131 executou o runner prospectivo offline da candidata congelada em
`docs/forex/fast131-nzdusd-donchian-fade-prospective-preregistration.md`.

Status:

`FAST131_OBSERVATION_NO_ELIGIBLE_TRADES_YET`

## Regra observada

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`

- par: `NZDUSD`;
- timeframe de sinal: `D1`;
- sinal: fade/contra rompimento Donchian de `100` dias;
- entrada: primeiro M1 disponível depois do fechamento D1;
- stop: `2.0 * ATR(14)` diário causal;
- alvo: `2R`;
- hold máximo: `20` dias;
- cenário primário de decisão: `STRESS`.

## Resultado da execução

- sinais reconstruídos: `79`;
- trades históricos materializados pela regra congelada: `65`;
- trades elegíveis pós-pré-registro: `0`;
- início do M1 local usado: `2019-01-01T22:02:00+00:00`;
- fim do M1 local usado: `2026-08-31T23:59:00+00:00`;
- timestamp de pré-registro: `2026-09-24T02:50:09-03:00`.

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
  `backend/scripts/research/run_fast131_nzdusd_donchian_fade_observation.py`;
- teste:
  `backend/tests/unit/test_fast131_nzdusd_donchian_fade_observation.py`;
- report JSON:
  `backend/data/lab_fast131/nzdusd-donchian-fade-observation/report.json`;
- sinais:
  `backend/data/lab_fast131/nzdusd-donchian-fade-observation/entry_signals.parquet`;
- trades materializados:
  `backend/data/lab_fast131/nzdusd-donchian-fade-observation/materialized_trade_rows.parquet`;
- observações:
  `backend/data/lab_fast131/nzdusd-donchian-fade-observation/observations.jsonl`.

Hashes:

- report:
  `b083d36d4d58b02508369dade28d15706225ed9d075c00859d76c78a49a21cc7`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades materializados:
  `059653bb49fcf5f17346e3750c327cd2e2f6f72e0bf76417703418a77dc457c7`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Interpretação

O estado correto continua sendo:

- temos uma candidata estatística pré-registrada;
- ainda não temos uma estratégia aprovada;
- ainda não há Testnet, Demo nem LIVE;
- o próximo avanço depende de importar/capturar M1 de NZDUSD posterior a
  `2026-09-24T02:50:09-03:00` e rerodar o Fast131.
