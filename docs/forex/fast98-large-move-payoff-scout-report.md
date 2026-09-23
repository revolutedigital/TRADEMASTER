# Fast98 — payoff scout de operações maiores

Resposta objetiva: operações maiores melhoraram o payoff bruto, mas ainda não
entregaram estratégia robusta.

Depois de Fast96/Fast97 enfraquecerem o caminho de gate para o candidato Fast95,
Fast98 testou a hipótese do Igor de buscar trades maiores, onde custo/spread pesa
menos no resultado.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Configuração

Runner:

`backend/scripts/research/run_book_payoff_model_scout.py`

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_payoff_model_scout.py \
  --wal-roots fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m \
  --output-root backend/data/microstructure_v1/reports/book-payoff-model-scout-fast98-large-move-10w-300s-100ms \
  --horizon-seconds 300 \
  --target-bps-values 10,15 \
  --trailing-distance-bps-values 10,15 \
  --top-fractions 0.05,0.10,0.20 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 8 \
  --minimum-executable-per-window 5
```

Grade:

- candidatos: `24`;
- modelos: `hist_gradient_payoff`, `ridge_payoff`;
- horizonte: `300s`;
- target / breakeven activation: `10`, `15` bps;
- trailing distance: `10`, `15` bps;
- top fraction: `5%`, `10%`, `20%`;
- latência: `100ms`;
- custo principal: `1 bps`;
- custo stress: `2 bps`.

## Status

`PAYOFF_MODEL_FAILED_REPLICATION`

Motivos:

- `no_one_position_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## Melhor candidato por total one-position

Pelo total líquido em one-position a `1 bps`, o melhor foi:

- modelo: `hist_gradient_payoff`;
- target: `10 bps`;
- trailing distance: `10 bps`;
- top fraction: `10%`;
- executions: `38`;
- net @ `1 bps`: `+213.60597563597887` bps;
- net @ `2 bps`: `+175.60597563597887` bps;
- positive windows @ `1 bps`: `8/10`;
- positive windows @ `2 bps`: `6/10`;
- janelas com mínimo de 5 execuções: `2/10`;
- pior janela @ `1 bps`: `-6.962554641148412` bps.

Esse candidato tem payoff alto, mas não é robusto: perde em fast16 e fast24,
fica frágil em fast22/fast28, e executa pouco por janela.

## Candidato mais robusto por janelas positivas

O candidato com mais janelas positivas foi:

- modelo: `hist_gradient_payoff`;
- target: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `5%`;
- executions: `28`;
- net @ `1 bps`: `+52.69230367972402` bps;
- net @ `2 bps`: `+24.69230367972402` bps;
- positive windows @ `1 bps`: `9/10`;
- positive windows @ `2 bps`: `8/10`;
- janelas com mínimo de 5 execuções: `2/10`;
- pior janela @ `1 bps`: `-8.93742656108234` bps.

Esse é o achado mais interessante do Fast98: trades maiores reduzem o efeito do
custo, mas deixam a amostra operacional pequena demais para aprovar.

## Leitura

Fast98 confirma parcialmente a hipótese de custo:

- quando o movimento alvo é maior, o payoff médio sobe;
- a sensibilidade a `2 bps` melhora;
- mas a estratégia fica rara no modo one-position;
- a robustez por janela ainda não fecha.

Isso não é estratégia ainda. É um próximo near-edge, agora na família de
operações maiores.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O próximo teste coerente é tentar um gate/seleção causal específico para o
candidato maior `target 10 / trailing 15 / top 5%`, ou ampliar a base de janelas
antes de promover qualquer hipótese.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/book-payoff-model-scout-fast98-large-move-10w-300s-100ms/report.json`;
- report sha256:
  `1eb65d2e26ac639078c47a4bc0f90fe209cc9790faa4b0b832a7c556a42e5267`.
