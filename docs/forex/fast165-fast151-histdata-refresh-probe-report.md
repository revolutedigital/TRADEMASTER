# Fast165 — Fast151 HistData refresh probe

Fast165 verifica o bloqueio `post_preregistration_m1_data_missing` do Fast163
sem sobrescrever a base canônica. Ele baixa setembro/2026 do HistData em área
isolada, normaliza para M1 e compara com os arquivos locais usados pelo
Fast151/Fast152.

Status:

`FAST165_HISTDATA_REFRESH_CONFIRMS_SOURCE_NOT_POST_PREREGISTRATION`

## Resposta curta

O bloqueio forward é real da fonte atual.

O refresh recém-baixado também termina em:

`2026-09-18T20:59:00+00:00`

Isso é antes do pré-registro Fast151:

`2026-09-24T06:18:19-03:00`

Portanto, não é problema de cache local. O HistData disponível agora ainda não
tem M1 pós-pré-registro para os quatro pares da cesta Fast151.

## Comparação por par

| Par | Local rows | Local end | Fresh rows | Fresh end |
|---|---:|---|---:|---|
| `AUDUSD` | `19,546` | `2026-09-18T20:59:00+00:00` | `19,546` | `2026-09-18T20:59:00+00:00` |
| `EURUSD` | `19,622` | `2026-09-18T20:59:00+00:00` | `19,622` | `2026-09-18T20:59:00+00:00` |
| `GBPUSD` | `19,646` | `2026-09-18T20:59:00+00:00` | `19,646` | `2026-09-18T20:59:00+00:00` |
| `NZDUSD` | `19,610` | `2026-09-18T20:59:00+00:00` | `19,610` | `2026-09-18T20:59:00+00:00` |

## Interpretação

Fast152 continua correto:

`FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA`

Fast165 adiciona uma evidência nova: atualizar a fonte HistData hoje não muda a
janela disponível. Então os caminhos reais são:

1. esperar o HistData publicar M1 depois do pré-registro Fast151; ou
2. importar uma fonte M1 independente que cubra depois de `2026-09-24`.

Sem isso, não existe observação prospectiva honesta da candidata Fast151.

## Segurança

Fast165 é apenas probe de dados:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- não sobrescreve a base canônica;
- não toca ledger;
- não toca corretora;
- não autoriza shadow, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast165_fast151_histdata_refresh_probe.py`;
- teste:
  `backend/tests/unit/test_fast165_fast151_histdata_refresh_probe.py`;
- report JSON gerado:
  `backend/data/lab_fast165/fast151-histdata-refresh-probe/report.json`;
- resumo:
  `backend/data/lab_fast165/fast151-histdata-refresh-probe/histdata-refresh-probe-summary.json`.

Hashes:

- report:
  `e5d090964e95670fb75ecfe09f266673079dfc4b0b77a6505268c153a89560a6`;
- resumo:
  `51d5b492afb1798458c4ea10f46185e152705b3f0cabe9020442f3cdfcb6501d`;
- runner:
  `3be6511eccc973384ebd453cf5497895149bd68235bbcbad0b2c11de9a452e71`;
- teste:
  `aab0c6a595e51f8025dc567fec15d06b6dd1fc402689edd343de91b02083ed06`.

## Validação

Comandos executados:

- `backend/.venv/bin/python -m pytest backend/tests/unit/test_fast165_fast151_histdata_refresh_probe.py -q --no-cov`;
- `cd backend && .venv/bin/ruff check scripts/research/run_fast165_fast151_histdata_refresh_probe.py tests/unit/test_fast165_fast151_histdata_refresh_probe.py`;
- `backend/.venv/bin/python backend/scripts/research/run_fast165_fast151_histdata_refresh_probe.py`;
- `backend/.venv/bin/python backend/scripts/research/run_fast152_fast151_readiness_monitor.py`.

Resultado:

- Fast165 tests: `3 passed`;
- Fast165 lint: `All checks passed`;
- Fast165 status:
  `FAST165_HISTDATA_REFRESH_CONFIRMS_SOURCE_NOT_POST_PREREGISTRATION`;
- Fast152 status:
  `FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA`.
