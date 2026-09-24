# Fast166 — Fast151 Dukascopy forward-source probe

Fast166 testa a alternativa pequena e independente depois do Fast165: usar
Dukascopy M1 somente como probe da janela forward da candidata Fast151, sem
sobrescrever a base canônica.

Status:

`FAST166_DUKASCOPY_SOURCE_ALIVE_BUT_PREREGISTRATION_DATE_UNAVAILABLE`

## Resposta curta

A Dukascopy está viva, mas ainda não entrega o dia do pré-registro Fast151.

Controle `2026-09-23`:

- 4 pares com dados;
- `5,741` minutos totais;
- todos terminam em `2026-09-23T23:59:00+00:00`.

Dia do pré-registro `2026-09-24`:

- 4 pares sem dados;
- cache total `0` bytes;
- nenhum minuto pós-pré-registro.

O pré-registro Fast151 foi em:

`2026-09-24T06:18:19-03:00`

Logo, a fonte alternativa pública confirma o mesmo bloqueio operacional: ainda
não existe janela M1 pós-pré-registro disponível para observar a candidata sem
vazar informação histórica.

## Comparação por data

| Data | Pares com rows | Total rows | Último minuto | Cache bytes | Leitura |
|---|---:|---:|---|---:|---|
| `2026-09-23` | `4/4` | `5,741` | `2026-09-23T23:59:00+00:00` | `87,142` | fonte viva |
| `2026-09-24` | `0/4` | `0` | — | `0` | dia ainda indisponível |

Detalhe do dia de controle:

| Par | Rows | Primeiro minuto | Último minuto |
|---|---:|---|---|
| `AUDUSD` | `1,437` | `2026-09-23T00:00:00+00:00` | `2026-09-23T23:59:00+00:00` |
| `EURUSD` | `1,437` | `2026-09-23T00:00:00+00:00` | `2026-09-23T23:59:00+00:00` |
| `GBPUSD` | `1,435` | `2026-09-23T00:00:00+00:00` | `2026-09-23T23:59:00+00:00` |
| `NZDUSD` | `1,432` | `2026-09-23T00:00:00+00:00` | `2026-09-23T23:59:00+00:00` |

## Interpretação

Fast165 já tinha mostrado que o HistData recém-baixado ainda para em
`2026-09-18T20:59:00+00:00`.

Fast166 fecha a hipótese “usa Dukascopy para pegar o forward agora”:

- a fonte responde e tem dados recentes;
- mas o dia `2026-09-24` ainda não está publicado;
- portanto não há overlay M1 independente suficiente para rerodar Fast151/Fast152
  com observação prospectiva.

Próximo caminho real:

1. esperar HistData ou Dukascopy publicar `2026-09-24`; ou
2. importar um arquivo M1 exportado de corretora/plataforma cobrindo depois de
   `2026-09-24T06:18:19-03:00`; e
3. só então montar overlay isolado e rerodar Fast151/Fast152.

## Segurança

Fast166 é apenas probe de fonte de dados:

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
  `backend/scripts/research/run_fast166_fast151_dukascopy_forward_probe.py`;
- teste:
  `backend/tests/unit/test_fast166_fast151_dukascopy_forward_probe.py`;
- report JSON gerado:
  `backend/data/lab_fast166/fast151-dukascopy-forward-probe/report.json`;
- resumo:
  `backend/data/lab_fast166/fast151-dukascopy-forward-probe/dukascopy-forward-probe-summary.json`.

Hashes:

- report:
  `618002e7967fe30cfc26e1527728d7d5478dfeecd7c18870d23dcdc07db66c6c`;
- resumo:
  `188a9a56a18f4bd582186830ecdb1e5c5b677cd42ec7936eefc8cfdb29411474`;
- runner:
  `2f9d4e3c091a065bae70cb0fe529b9c49420d4b53072a61fc1ffd1b4a77a5243`;
- teste:
  `761bb6d8c79b995ec09759e0d03d31afc854bf12f3369460c57fd6d756a00cf0`.

## Validação

Comandos executados:

- `backend/.venv/bin/python -m pytest backend/tests/unit/test_fast166_fast151_dukascopy_forward_probe.py -q --no-cov`;
- `cd backend && .venv/bin/ruff check scripts/research/run_fast166_fast151_dukascopy_forward_probe.py tests/unit/test_fast166_fast151_dukascopy_forward_probe.py`;
- `backend/.venv/bin/python backend/scripts/research/run_fast166_fast151_dukascopy_forward_probe.py`.

Resultado:

- Fast166 tests: `3 passed`;
- Fast166 lint: `All checks passed`;
- Fast166 status:
  `FAST166_DUKASCOPY_SOURCE_ALIVE_BUT_PREREGISTRATION_DATE_UNAVAILABLE`.
