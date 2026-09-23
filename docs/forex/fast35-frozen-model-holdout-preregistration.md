# Fast35 — pré-registro de holdout prospectivo do candidato Fast34

Data de início: 2026-09-23  
Mercado: BTCUSDT futures  
Janela prospectiva: `backend/data/microstructure_v1/prospective-wal-fast35-30m`  
Status: pré-registrado antes da liquidação do resultado.

## Hipótese congelada

Avaliar o melhor candidato de pesquisa encontrado em Fast34 sem alterar
parâmetros:

- modelo: `hist_gradient`;
- treino: Fast24 + Fast28 + Fast29 + Fast32;
- holdout: Fast35;
- horizonte: 120s;
- alvo de ativação / breakeven: 8 bps;
- trailing distance: 3 bps;
- top-p: 5%;
- latência simulada: 100ms;
- stop inicial: 20 bps;
- decisão: a cada 5s;
- custo explícito principal: 1 bps;
- custo de estresse: 2 bps.

## Comando pré-registrado

```bash
backend/.venv/bin/python backend/scripts/research/run_book_frozen_model_holdout.py \
  --training-wal-roots fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m \
  --holdout-wal-root fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m \
  --output-root backend/data/microstructure_v1/reports/book-frozen-model-holdout-fast35-100ms \
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
  --minimum-executable-count 5
```

## Critério de leitura

Este holdout não pode aprovar estratégia sozinho.

Leitura mínima para manter a hipótese viva:

- `one_position_summary.executable_count >= 5`;
- `one_position` líquido positivo a 1 bps;
- `independent` líquido positivo a 1 bps;
- sem mudança de parâmetros depois de ver Fast35.

Leitura forte:

- critérios acima;
- `one_position` e `independent` positivos também a 2 bps;
- drawdown aceitável;
- amostra suficiente para repetir em nova janela.

Se falhar no custo de 1 bps ou por amostra mínima, o candidato Fast34 é
rejeitado como caminho principal.

## Segurança

Pesquisa apenas:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- sem Testnet/LIVE.
