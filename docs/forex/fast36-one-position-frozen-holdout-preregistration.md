# Fast36 — pré-registro da tese one-position-first

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Status: pré-registrado antes da captura/liquidação Fast36.

## Por que esta rodada existe

Fast35 reprovou o critério estrito porque o modo independente ficou negativo
em 1 bps. Porém, o modo operacional de uma posição por vez foi forte:

- 10 execuções;
- net +4,4421 bps em 1 bps de custo;
- net +3,4421 bps em 2 bps de custo;
- 80% de win rate;
- 7 de 10 saídas por `TRAILING_STOP`.

Isso justifica uma nova tese, mas não autoriza mudar a leitura de Fast35
retroativamente.

## Hipótese congelada Fast36

Avaliar a mesma entrada/modelo do candidato Fast34/Fast35, mas com gate de
aprovação de pesquisa focado no modo que seria realmente operado:
`one_position`.

Parâmetros fixos:

- modelo: `hist_gradient`;
- treino: Fast24 + Fast28 + Fast29 + Fast32 + Fast35;
- holdout: Fast36;
- horizonte: 120s;
- target / breakeven / ativação do trailing: 8 bps;
- trailing distance: 3 bps;
- top-p: 5%;
- latência simulada: 100ms;
- stop inicial: 20 bps;
- decisão: a cada 5s;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- minimum executable count: 5;
- required positive modes: `one_position`.

## Comando pré-registrado

```bash
backend/.venv/bin/python backend/scripts/research/run_book_frozen_model_holdout.py \
  --training-wal-roots fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m \
  --holdout-wal-root fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m \
  --output-root backend/data/microstructure_v1/reports/book-frozen-model-holdout-fast36-one-position-100ms \
  --model hist_gradient \
  --horizon-seconds 120 \
  --target-bps 8 \
  --top-fraction 0.05 \
  --trailing-distance-bps 3 \
  --decision-stride-seconds 5 \
  --max-book-staleness-ms 1000 \
  --latency-ms 100 \
  --initial-stop-bps 20 \
  --cost-levels-bps 0,0.25,0.5,1,2 \
  --profit-cost-bps 1 \
  --stress-cost-bps 2 \
  --minimum-executable-count 5 \
  --required-positive-modes one_position
```

## Critério de leitura

Este holdout também não aprova estratégia sozinho.

Para manter a tese viva:

- `one_position_summary.executable_count >= 5`;
- `one_position` líquido positivo a 1 bps;
- sem alteração de parâmetro depois de ver Fast36.

Leitura forte:

- critérios acima;
- `one_position` também positivo a 2 bps;
- drawdown controlado;
- saídas por trailing presentes em quantidade relevante.

O modo independente será reportado, mas não será gate da tese Fast36, porque a
política candidata é explicitamente portfolio-aware/uma-posição.

## Segurança

Pesquisa apenas:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- sem Testnet/LIVE.
