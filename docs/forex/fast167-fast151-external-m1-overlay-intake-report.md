# Fast167 — Fast151 external M1 overlay intake

Fast167 prepara o caminho seguro para destravar o bloqueio forward quando houver
um CSV M1 exportado de corretora/plataforma. Ele não sobrescreve a base canônica:
normaliza o CSV externo, monta um overlay isolado e só então pode rerodar
Fast151/Fast152 em cima desse overlay.

Status:

`FAST167_EXTERNAL_M1_INPUT_PENDING`

## Resposta curta

O pipeline de entrada externa está pronto. Falta o arquivo M1 externo.

Template gerado:

`backend/data/lab_fast167/fast151-external-m1-overlay-intake/external-m1-normalized-template.csv`

Colunas exigidas:

- `pair`;
- `timestamp`;
- `bid_open`;
- `bid_high`;
- `bid_low`;
- `bid_close`;
- `ask_open`;
- `ask_high`;
- `ask_low`;
- `ask_close`.

O timestamp precisa estar em UTC. O exemplo já começa em:

`2026-09-24T09:19:00+00:00`

Isso é o primeiro minuto depois do pré-registro Fast151:

`2026-09-24T06:18:19-03:00`

## Como usar

Preencher um CSV com M1 bid/ask OHLC para os 4 pares:

- `AUDUSD`;
- `EURUSD`;
- `GBPUSD`;
- `NZDUSD`.

Depois rodar:

`backend/.venv/bin/python backend/scripts/research/run_fast167_fast151_external_m1_overlay_intake.py --external-m1-path <m1-externo.csv>`

O Fast167 então:

1. valida colunas e timestamps;
2. rejeita bid/ask cruzado;
3. rejeita OHLC inconsistente;
4. grava Parquet normalizado por par;
5. monta `overlay_m1` com symlinks/cópias dos arquivos canônicos e externos;
6. se todos os 4 pares tiverem linhas pós-pré-registro, roda Fast152 nesse overlay.

## Por que isso importa

Fast165 mostrou que o HistData ainda para em `2026-09-18T20:59:00+00:00`.

Fast166 mostrou que a Dukascopy está viva em `2026-09-23`, mas ainda não entrega
`2026-09-24`.

Fast167 fecha o próximo caminho prático: se vier um export de corretora ou
plataforma com M1 pós-pré-registro, o projeto já sabe ingerir, isolar e testar
sem contaminar a base histórica.

## Segurança

Fast167 é apenas intake/overlay de dados:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- não sobrescreve M1 canônico;
- não toca corretora;
- não toca ledger;
- não autoriza shadow, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast167_fast151_external_m1_overlay_intake.py`;
- teste:
  `backend/tests/unit/test_fast167_fast151_external_m1_overlay_intake.py`;
- report JSON gerado:
  `backend/data/lab_fast167/fast151-external-m1-overlay-intake/report.json`;
- resumo:
  `backend/data/lab_fast167/fast151-external-m1-overlay-intake/external-m1-overlay-intake-summary.json`;
- template:
  `backend/data/lab_fast167/fast151-external-m1-overlay-intake/external-m1-normalized-template.csv`.

Hashes:

- report:
  `5d51c6821b528ae840fb4ff3742162de7f5afd1bf36a859bb8b3f5d847d7007d`;
- resumo:
  `ce5b78e284750b6c129fecc337c2865948467f480cc4310f690aac68c3c9d2f7`;
- template:
  `95ae562c76d110131dadbb3559e1c8ba88e54b72128bffbcd98e4fd320df4e18`;
- runner:
  `b3a2139e469bb34b2769d7c3b23f0c72c21072f4b1c0c4f238748be894f91d7c`;
- teste:
  `68b88f46ac74cfce8715950bc3eb52378ae2ca3d29eb1e8a2f27e36e5aa2a44c`.

## Validação

Comandos executados:

- `backend/.venv/bin/python -m pytest backend/tests/unit/test_fast167_fast151_external_m1_overlay_intake.py -q --no-cov`;
- `cd backend && .venv/bin/ruff check scripts/research/run_fast167_fast151_external_m1_overlay_intake.py tests/unit/test_fast167_fast151_external_m1_overlay_intake.py`;
- `backend/.venv/bin/python backend/scripts/research/run_fast167_fast151_external_m1_overlay_intake.py`.

Resultado:

- Fast167 tests: `5 passed`;
- Fast167 lint: `All checks passed`;
- Fast167 status:
  `FAST167_EXTERNAL_M1_INPUT_PENDING`.
