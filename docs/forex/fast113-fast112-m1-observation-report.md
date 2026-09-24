# Fast113 — materializador M1 para observação Fast112

Resposta objetiva: ainda não temos estratégia. Agora temos um observador melhor:
ele reconstrói a hipótese Fast112 direto do M1 bruto local, em vez de depender
da tabela histórica materializada pelo Fast111.

## O que foi testado

- par: `USDJPY`;
- configuração congelada: `F1a.rr2.5.w0.6-2.0.se11:00.x16:30`;
- filtro congelado: `london_entry_hour=8 && stop_pips>94.88491666666664`;
- janela prospectiva: `entry_time > 2026-09-24T02:59:14Z`;
- base de dados: `backend/data/raw/fx_m1_hd`;
- motor: `core.run_simulation` + `prepare_run` + `finalize_trades`;
- cenários: `fusion_zero` e `stress`.

## Safety boundary

Fast113 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE.

## Resultado

Fonte M1 local:

- início: `2019-01-01T22:00:00+00:00`;
- fim: `2026-09-18T20:59:00+00:00`;
- linhas M1: `2,819,500`.

Validação do último download HistData:

- arquivo: `backend/data/raw/fx_m1_hd/USDJPY_validation.csv`;
- hash: `16a1f265ff961afa4df0a6959dd644aacd611e0e3f7e1dcdc125ca2c4d4f2dd1`;
- mês reportado: `2026-09`;
- `passes=false`;
- `hours_compared=0`;
- motivo prático: ainda não há oracle H1 local cobrindo o mês baixado, então
  setembro/2026 entra como dado observacional não-validado, não como evidência
  para aprovar estratégia.

Materialização:

- trades reconstruídos da regra congelada: `534`;
- observações elegíveis pós-pré-registro: `0`;
- status: `FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET`;
- motivo principal: o M1 local termina em `2026-09-18`, antes do pré-registro em
  `2026-09-24T02:59:14Z`.

Hashes:

- report:
  `fc6e5a5cb52e78915c9a2a9c82c5f3c45ba334b2f768a381d929d8cedf79f897`;
- materialized trade rows:
  `33deb104d38cc9097e92f60e559fd2758f653e7cbb1f0cf17cdc382b31f9a6b8`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Arquivos

- runner:
  `backend/scripts/research/run_fast113_fast112_m1_observation.py`;
- teste:
  `backend/tests/unit/test_fast113_fast112_m1_observation.py`;
- relatório JSON local:
  `backend/data/lab_fast113/fast112-m1-observation/report.json`;
- trades materializados locais:
  `backend/data/lab_fast113/fast112-m1-observation/materialized_trade_rows.parquet`;
- observações locais:
  `backend/data/lab_fast113/fast112-m1-observation/observations.jsonl`.

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast112_usdjpy_breakout_observation.py \
  backend/tests/unit/test_fast113_fast112_m1_observation.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast112_usdjpy_breakout_observation.py \
  backend/scripts/research/run_fast113_fast112_m1_observation.py \
  backend/tests/unit/test_fast112_usdjpy_breakout_observation.py \
  backend/tests/unit/test_fast113_fast112_m1_observation.py

PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast113_fast112_m1_observation.py
```

Resultado:

- unit tests: `8 passed`;
- lint: `All checks passed`;
- materializador: sucesso.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast113 prova que o encanamento consegue reconstruir a hipótese congelada direto
do M1 bruto. O próximo passo útil é alimentar M1 posterior a
`2026-09-24T02:59:14Z` e reexecutar o mesmo runner sem recalibrar a regra.
