# Fast124 — USDJPY compression breakout prospective observation run

Fast124 congelou a melhor pista atual (`USDJPY` H1 compression breakout) e
rodou uma avaliação prospectiva offline. O corte prospectivo é o timestamp do
commit de pré-registro:

- `preregistration_commit`:
  `0a0fb2183ecd3de19b10d2832230194133afe79d`;
- `preregistration_timestamp`: `2026-09-24T01:50:12-03:00`;
- regra congelada: `tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r`.

## Resultado

Status:

`FAST124_OBSERVATION_NO_ELIGIBLE_TRADES_YET`

O runner reconstruiu a regra a partir de M1 bid/ask local, mas não há trades
elegíveis depois do pré-registro porque o dado local termina antes do corte.

Números do run:

- M1 local: `2,819,500` linhas;
- primeira barra M1: `2019-01-01T22:00:00+00:00`;
- última barra M1: `2026-09-18T20:59:00+00:00`;
- trades materializados pela regra congelada em todo o histórico local: `959`;
- observações prospectivas elegíveis: `0`;
- observações JSONL: vazio
  (`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

## Segurança

O Fast124 permanece estritamente offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Validação do dado

O CSV de validação local existe, mas o mês reportado (`2026-09`) aparece como
não validado contra oracle:

- `failed_reported_months`: `["2026-09"]`;
- `zero_oracle_hour_months`: `["2026-09"]`.

Isso não afeta a conclusão prospectiva imediata, porque não há nenhuma entrada
após o pré-registro. Para observar Fast124 de verdade, o próximo dado M1 precisa
incluir barras posteriores a `2026-09-24T01:50:12-03:00`.

## Artefatos

- runner:
  `backend/scripts/research/run_fast124_usdjpy_compression_observation.py`;
- teste:
  `backend/tests/unit/test_fast124_usdjpy_compression_observation.py`;
- relatório JSON:
  `backend/data/lab_fast124/usdjpy-compression-breakout-observation/report.json`;
- trades materializados:
  `backend/data/lab_fast124/usdjpy-compression-breakout-observation/materialized_trade_rows.parquet`;
- observações prospectivas:
  `backend/data/lab_fast124/usdjpy-compression-breakout-observation/observations.jsonl`.

Hashes:

- report:
  `9c9856ab52746f4babc2c31095e41c9e4ffd220b4482701d7592a623d8c32780`;
- materialized trade rows:
  `539117a69469c05de4c6414db8a30a755bbda7919597200516c2266f68e85e03`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Interpretação

Ainda não temos estratégia operável. Temos uma hipótese congelada e pronta para
ser observada em dados futuros.

A pergunta agora não é mais "qual parâmetro melhora o backtest?". A pergunta
certa ficou objetiva:

> depois de `2026-09-24T01:50:12-03:00`, essa regra continua gerando trades
> com lucro líquido em cenário stress?

Enquanto essa resposta não existir em dados novos, o status correto é aguardar
observação prospectiva.
