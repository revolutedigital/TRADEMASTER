# Fast58 — sensibilidade de custo e latência do Fast57

Data: 2026-09-23  
Status: auditoria implementada e rodada; candidato segue promissor, mas
economicamente frágil.  
Escopo: research-only.

## Pergunta da rodada

Fast57 ficou positivo com custo stress de 2 bps. A pergunta do Fast58 foi:

> se o custo/slippage/latência real for pior, o edge ainda existe?

Esta rodada não escolhe novo modelo, não muda threshold, não muda gate e não muda
trailing. Ela só reexecuta o candidato congelado em uma grade de execução.

Script:

`backend/scripts/research/run_fast58_cost_latency_sensitivity.py`

Flags de segurança mantidas:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Candidato auditado

- modelo: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- threshold: `0,90195311025804`;
- top-p: 1%;
- horizonte: 300 s;
- política: `micro_8_3`;
- gate: `recent_300s_lte_0`;
- grade de latência: 100, 250, 500 e 1000 ms;
- grade de custo stress: 2, 3, 4, 5 e 8 bps roundtrip.

## Resultado no seed novo `2026-09-21` a `2026-09-22`

Comando:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_fast58_cost_latency_sensitivity.py
```

Relatório local ignorado pelo git:

`backend/data/microstructure_v1/reports/fast58-cost-latency-sensitivity/report.json`

Hash:

`f3ed37fcdb93ff56440c56a9e857bd2cb7b5c5794f5a1a1e92c47733b842707c`

Resumo:

| Latência | Break-even custo roundtrip | Maior custo testado ainda positivo | Cenários positivos |
| --- | ---: | ---: | ---: |
| 100 ms | 4,4290 bps | 4 bps | 3/5 |
| 250 ms | 4,1436 bps | 4 bps | 3/5 |
| 500 ms | 4,3056 bps | 4 bps | 3/5 |
| 1000 ms | 4,3265 bps | 4 bps | 3/5 |

Leitura:

- o gate bateu o sem-gate em 20/20 cenários;
- o candidato ficou positivo em 12/20 cenários;
- quebrou quando o custo stress subiu para 5 e 8 bps;
- status: `FAST58_SENSITIVITY_FRAGILE_RESEARCH_ONLY`.

## Diagnóstico histórico `Fast43` a `Fast53`

Comando:

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_fast58_cost_latency_sensitivity.py \
  --blocks \
    fast43=backend/data/microstructure_v1/research-lowcost-fast42-holdout \
    fast44=backend/data/microstructure_v1/research-lowcost-fast44-holdout2 \
    fast45=backend/data/microstructure_v1/research-lowcost-fast45-holdout3 \
    fast46=backend/data/microstructure_v1/research-lowcost-fast46-holdout \
    fast47=backend/data/microstructure_v1/research-lowcost-fast47-holdout \
    fast48=backend/data/microstructure_v1/research-lowcost-fast48-holdout \
    fast49=backend/data/microstructure_v1/research-lowcost-fast49-holdout \
    fast50=backend/data/microstructure_v1/research-lowcost-fast50-holdout \
    fast51=backend/data/microstructure_v1/research-lowcost-fast51-holdout \
    fast52=backend/data/microstructure_v1/research-lowcost-fast52-holdout \
    fast53=backend/data/microstructure_v1/research-lowcost-fast53-holdout \
  --output-root backend/data/microstructure_v1/reports/fast58-cost-latency-sensitivity-fast43-fast53-diagnostic
```

Relatório local ignorado pelo git:

`backend/data/microstructure_v1/reports/fast58-cost-latency-sensitivity-fast43-fast53-diagnostic/report.json`

Hash:

`64392f0ca9afcbdf53bd8b159f0a1cdad9b659cabed42b219ec56b6ec842fdd7`

Resumo:

| Latência | Break-even custo roundtrip | Maior custo testado ainda positivo | Cenários positivos |
| --- | ---: | ---: | ---: |
| 100 ms | 2,6668 bps | 2 bps | 1/5 |
| 250 ms | 2,6211 bps | 2 bps | 1/5 |
| 500 ms | 2,1419 bps | 2 bps | 1/5 |
| 1000 ms | 2,2121 bps | 2 bps | 1/5 |

Leitura:

- o gate bateu o sem-gate em 20/20 cenários;
- o candidato ficou positivo em só 4/20 cenários;
- acima de 2 bps, o histórico fica negativo;
- status: `FAST58_SENSITIVITY_FRAGILE_RESEARCH_ONLY`.

## Decisão

Ainda não há estratégia aprovada.

O Fast58 reforça duas coisas ao mesmo tempo:

1. o gate `5min≤0` está fazendo trabalho real, porque bate o sem-gate em todos
   os cenários testados;
2. a margem econômica é estreita demais para ignorar execução. No histórico,
   o break-even fica perto de 2,1 a 2,7 bps roundtrip.

Consequência prática:

- Fast57 ainda deve continuar acumulando dias prospectivos;
- se Fast57 passar, o próximo gate precisa medir custo de execução real por
  book/shadow antes de qualquer Testnet;
- esse candidato só faz sentido em ambiente de custo muito baixo, fill bom e
  spread/slippage controlados.

Sem Testnet/LIVE.
