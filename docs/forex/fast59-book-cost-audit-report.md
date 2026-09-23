# Fast59 — auditoria de custo top-of-book dos WALs

Data: 2026-09-23  
Status: auditoria implementada e rodada nos WALs prospectivos locais; amostra
boa para spread, parcial para freshness.  
Escopo: research-only.

## Pergunta da rodada

Fast58 mostrou que o candidato Fast57 é frágil acima de poucos bps de custo
roundtrip. Fast59 pergunta:

> o top-of-book capturado nos WALs locais tem spread/freshness compatível com um
> edge de baixo custo?

Esta rodada não avalia sinal, não aprova estratégia e não envia ordem. Ela mede
somente a condição de execução observada no book capturado.

Script:

`backend/scripts/research/run_fast59_book_cost_audit.py`

Flags de segurança:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Comando validado

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_fast59_book_cost_audit.py
```

O comando descobre, por padrão, os WALs `prospective-wal-fast*` com stream
`depth`.

## Thresholds usados

| Métrica | Threshold |
| --- | ---: |
| P95 custo roundtrip top-of-book | <= 2,0 bps |
| P95 receive delay | <= 1000 ms |
| P05 notional no topo por lado | >= US$ 10.000 |
| Mínimo de eventos depth | >= 100 |

Observação: o custo roundtrip top-of-book aqui é só o spread bid/ask aproximado
por `log(ask/bid) * 10000`. Não inclui taxa da corretora, fila, impacto,
slippage de tamanho, rejeição, nem latência de execução.

## Resultado

Relatório local ignorado pelo git:

`backend/data/microstructure_v1/reports/fast59-book-cost-audit/report.json`

Hash:

`2d9443e9cb404e8b57d0e2055bafd84d86285b12a8d3a56532ad014788ba573e`

Resumo:

| Métrica | Valor |
| --- | ---: |
| Partições auditadas | 10 |
| Partições elegíveis | 9 |
| Datas UTC únicas | 1 |
| Máximo P95 custo roundtrip top-of-book | 0,01185 bps |
| Máximo P95 receive delay | 4431,701 ms |
| Menor P05 notional topo por lado | US$ 32.247,23 |

Status:

`FAST59_BOOK_COST_SAMPLE_PARTIAL_RESEARCH_ONLY`

Motivos:

- `eligible_partitions:9/10`;
- `do_not_promote_to_strategy`.

## Partição que falhou

`prospective-wal-fast36-30m`, em `2026-09-23`.

Motivo:

- `p95_receive_delay_above_threshold:4431.701>1000.0`.

Mesmo nessa partição, o spread P95 foi baixo:

- P95 custo roundtrip top-of-book: `0,01185 bps`;
- P05 bid notional: `US$ 43.862,24`;
- P05 ask notional: `US$ 32.247,23`.

## Decisão

Ainda não há estratégia aprovada.

O achado é bom e ruim ao mesmo tempo:

1. Bom: o spread top-of-book observado não parece ser o custo que mata o edge.
   O P95 ficou muito abaixo de 2 bps nos WALs locais.
2. Ruim: a captura precisa ser fiscalizada por freshness. Uma das 10 partições
   teve P95 de receive delay acima de 4,4 s, o que não serve como evidência de
   execução fina.

Consequência prática:

- continuar Fast57 acumulando dias prospectivos;
- continuar Fast59/Book Evidence até chegar em 60 dias completos, se um
  candidato sobreviver;
- medir taxa/fill/slippage real em shadow antes de Testnet, porque o book spread
  sozinho não representa custo total;
- qualquer partição com freshness ruim deve ser excluída ou marcada como
  inelegível para evidência.

Sem Testnet/LIVE.
