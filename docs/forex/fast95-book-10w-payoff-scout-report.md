# Fast95 — book payoff scout com 10 janelas

Resposta objetiva: ainda não temos estratégia aprovada, mas este foi o melhor
sinal desde Fast35.

Fast95 voltou para a família com features completas de livro (`depth`, spot,
mark/index, basis, microprice, pressure imbalance etc.) e ampliou a validação
para 10 janelas:

- Fast16;
- Fast20;
- Fast21;
- Fast22;
- Fast24;
- Fast28;
- Fast29;
- Fast32;
- Fast35;
- Fast36.

O teste reaproveitou o runner existente
`backend/scripts/research/run_book_payoff_model_scout.py`.

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

## Por que este teste importa

Fast93 e Fast94 mostraram que o universo Fast92, com vetor causal reduzido, não
tem edge estável. Antes de concluir que "não existe nada", Fast95 testa a fonte
mais rica que já temos localmente: os WALs com livro completo.

Isso muda a pergunta:

> O problema é falta de edge, ou falta de features de microestrutura/livro?

## Configuração

Comando principal:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_payoff_model_scout.py \
  --wal-roots fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m \
  --output-root backend/data/microstructure_v1/reports/book-payoff-model-scout-fast95-10w-120s-100ms \
  --horizon-seconds 120 \
  --target-bps-values 3,5,8 \
  --trailing-distance-bps-values 3,5,8 \
  --top-fractions 0.02,0.05,0.10,0.20 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 8 \
  --minimum-executable-per-window 5
```

Grade:

- candidatos: `72`;
- modelos: `hist_gradient_payoff`, `ridge_payoff`;
- horizonte: `120s`;
- target / breakeven activation: `3`, `5`, `8` bps;
- trailing distance: `3`, `5`, `8` bps;
- top fraction: `2%`, `5%`, `10%`, `20%`;
- latência: `100ms`;
- custo principal: `1 bps`;
- custo stress: `2 bps`.

## Status

`PAYOFF_MODEL_FAILED_REPLICATION`

Motivos:

- `no_one_position_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## Melhor candidato operacional

Ordenando por janelas positivas no modo uma-posição e net total a 1 bps:

- modelo: `hist_gradient_payoff`;
- target: `3 bps`;
- trailing distance: `8 bps`;
- top fraction: `10%`;
- selected signals: `620`;
- feature count: `122`.

### Resultado independente

| Métrica | Valor |
|---|---:|
| Executions | `620` |
| Gross total | `+1573.6274512438722` bps |
| Net total 1 bps | `+953.6274512438722` bps |
| Net total 2 bps | `+333.6274512438722` bps |
| Positive windows @ 1 bps | `8/10` |
| Positive windows @ 2 bps | `7/10` |
| Minimum execution window count | `10` |
| Worst window net @ 1 bps | `-2.360841401202344` bps |
| Exits | `105 BREAKEVEN_STOP`, `6 INITIAL_STOP`, `436 TIMEOUT`, `73 TRAILING_STOP` |

### Resultado one-position

| Métrica | Valor |
|---|---:|
| Executions | `90` |
| Gross total | `+227.93733731018628` bps |
| Net total 1 bps | `+137.93733731018628` bps |
| Net total 2 bps | `+47.93733731018628` bps |
| Positive windows @ 1 bps | `9/10` |
| Positive windows @ 2 bps | `5/10` |
| Minimum execution window count | `9` |
| Worst window net @ 1 bps | `-1.865852022264431` bps |
| Exits | `16 BREAKEVEN_STOP`, `3 INITIAL_STOP`, `59 TIMEOUT`, `12 TRAILING_STOP` |

Por janela no modo one-position a 1 bps:

| Janela | Execuções | Net 1 bps |
|---|---:|---:|
| Fast16 | 5 | `+0.08947834417082934` |
| Fast20 | 9 | `+15.976280292806141` |
| Fast21 | 11 | `+13.826133244174503` |
| Fast22 | 5 | `-9.329260111322156` |
| Fast24 | 7 | `+23.496676313367075` |
| Fast28 | 11 | `+1.0654715163801551` |
| Fast29 | 12 | `+14.229140519778767` |
| Fast32 | 3 | `+1.3740784385281462` |
| Fast35 | 12 | `+10.886010337945699` |
| Fast36 | 15 | `+66.3233284143571` |

## Baseline aninhado sem gate

Também rodei o runner de regime/no-trade em modo `keep_fraction=1.00`, que é
equivalente a "sem gate", mas pelo caminho aninhado do scout de regime.

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_regime_gate_payoff_scout.py \
  --wal-roots fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m \
  --output-root backend/data/microstructure_v1/reports/book-regime-gate-payoff-scout-fast95-10w-target3-trail8-top10-nogate-100ms \
  --model hist_gradient_payoff \
  --horizon-seconds 120 \
  --target-bps 3 \
  --trailing-distance-bps 8 \
  --top-fraction 0.10 \
  --gate-keep-fractions 1.00 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 8 \
  --minimum-executable-per-window 5
```

Resultado:

- status: `REGIME_GATE_PAYOFF_FAILED_REPLICATION`;
- one-position net 1 bps: `+137.93733731018628` bps;
- one-position positive windows @ 1 bps: `9/10`;
- independent net 1 bps: `+953.6274512438722` bps;
- independent positive windows @ 1 bps: `8/10`.

O scout completo de gates com `0.50,0.70,0.85,1.00` foi interrompido após
ficar caro demais para esta rodada interativa. O gargalo foi avaliação de gates
em Pandas, não ausência de dado. Isso deve virar runner otimizado se Fast95 for
evoluído para Fast96.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Apesar do resultado ser o melhor sinal recente, ainda falta:

1. replicar em todas as janelas a 1 bps no modo operacional;
2. garantir mínimo de execuções por janela;
3. sobreviver a 2 bps de stress com menos fragilidade;
4. testar um gate de regime otimizado sem escolher olhando Fast22;
5. rodar uma janela prospectiva nova congelada se o gate sobreviver.

## O que mudou na leitura

Fast95 muda o diagnóstico:

- o vetor causal reduzido Fast92 parece fraco;
- features completas de livro trazem sinal material;
- existe um quase-edge: `9/10` janelas positivas one-position a 1 bps;
- a falha está concentrada em regime/janela, não em ausência total de sinal.

Isso é diferente de Fast93/Fast94. O caminho mais promissor agora é:

> otimizar e pré-registrar um regime/no-trade gate para esse candidato Fast95,
> sem olhar o holdout, e depois confirmar em janela prospectiva nova.

## Arquivos e hashes

- Fast95 payoff report:
  `e8726678d6d7ed2a6d52ac7301093cc8ec4af461f842441e4eef1c6b8ed8c309`;
- Fast95 no-gate nested report:
  `8d9b70a0f768c64fe82008ff204a79ec5819f5449e6abc6cfd00e88e1618a5a2`.
