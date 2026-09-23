# Fast60 — shadow dry-run live-like do Fast57

Data: 2026-09-23  
Status: dry-run implementado e rodado em WALs prospectivos locais; o candidato
gera sinais gated em ambiente live-like.  
Escopo: research-only.

## Pergunta da rodada

Depois do Fast58 e Fast59, a pergunta virou:

> o candidato Fast57 realmente emitiria sinais em uma captura live-like, com
> features causais e o mesmo gate `5min≤0`?

Fast60 lê WALs cronológicos, calcula features online, pontua o artefato
congelado e aplica o gate de densidade. Ele não grava ledger, não envia ordem e
não liquida resultado.

Script:

`backend/scripts/research/run_fast60_fast57_shadow_dry_run.py`

Flags de segurança:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`;
- `committed=false`;
- `dry_run=true`.

## Correção importante de execução

As decisões são alinhadas ao próximo múltiplo UTC do stride de 5 segundos. Isso
evita decisões em milissegundos arbitrários derivados do primeiro trade do WAL.

## Comando validado

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_fast60_fast57_shadow_dry_run.py
```

Por padrão, o comando descobre `prospective-wal-fast*`, usa o artefato Fast57 e
aplica:

- top-p: 1%;
- threshold: `0,90195311025804`;
- horizonte: 300 s;
- gate: `recent_signal_count <= 0` nos últimos 300 s;
- stride: 5 s;
- warmup: 60 s.

## Resultado

Relatório local ignorado pelo git:

`backend/data/microstructure_v1/reports/fast60-fast57-shadow-dry-run/report.json`

Hash:

`a072661a8b3834da42f2d63b8e941eb45b68ab3e78cfaabb14bda5dee82bb4a6`

Decisões top-p brutas:

`backend/data/microstructure_v1/reports/fast60-fast57-shadow-dry-run/raw_top_p_decisions.jsonl`

Hash:

`d59d7b1695f7c7e1cf754bb9e53284f727a19932146b8890e26233be77b23cd9`

Decisões após gate:

`backend/data/microstructure_v1/reports/fast60-fast57-shadow-dry-run/gated_shadow_decisions.jsonl`

Hash:

`d4907d1bf30973086c29a25ee3e22cde602af2ba2634d6f5bf249c926f8efc9b`

Resumo:

| Métrica | Valor |
| --- | ---: |
| WAL roots avaliados | 10 |
| Runs com sinais gated | 8 |
| Sinais top-p brutos | 550 |
| Sinais após gate `5min≤0` | 11 |

Status:

`FAST60_GATED_SHADOW_SIGNALS_RESEARCH_ONLY`

Motivos:

- `gated_signals:11`;
- `not_approved_strategy`;
- `shadow_outcomes_required`.

## Resultado por janela

| WAL | Top-p bruto | Após gate |
| --- | ---: | ---: |
| prospective-wal-fast16-18m | 0 | 0 |
| prospective-wal-fast20-30m | 16 | 1 |
| prospective-wal-fast21-30m | 38 | 1 |
| prospective-wal-fast22-30m | 2 | 1 |
| prospective-wal-fast24-30m | 22 | 2 |
| prospective-wal-fast28-30m | 13 | 1 |
| prospective-wal-fast29-30m | 51 | 2 |
| prospective-wal-fast32-30m | 0 | 0 |
| prospective-wal-fast35-30m | 71 | 1 |
| prospective-wal-fast36-30m | 337 | 2 |

## Qualidade do book nos 11 sinais gated

Todos os 11 sinais gated tiveram book disponível no momento da decisão.

Observado nas decisões gated:

- `book_age_ms`: 32 ms a 99 ms;
- `spread_bps`: aproximadamente 0,01146 a 0,01170 bps;
- menor notional no topo entre os lados: US$ 11.292,35.

Isso é compatível com a leitura do Fast59: spread top-of-book não parece ser o
problema principal nas capturas boas. A incerteza continua sendo fill, taxa,
fila, slippage e outcome real.

## Decisão

Ainda não há estratégia aprovada.

Fast60 mostra que:

1. o candidato não é só uma curiosidade offline; ele emite sinais em WALs
   live-like;
2. o gate `5min≤0` reduz agressivamente a exposição: 550 top-p brutos viraram
   11 sinais;
3. esses sinais gated aconteceram com book fresco e spread baixo;
4. falta settlement/outcome de shadow para saber se esses 11 sinais teriam dado
   dinheiro.

Próximo gate correto:

1. liquidar esses sinais com replay/executable settlement;
2. repetir em janela prospectiva maior;
3. só depois abrir qualquer discussão de Testnet.

Sem Testnet/LIVE.
