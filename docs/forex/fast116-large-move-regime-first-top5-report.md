# Fast116 — regime-first no large-move top 5%

Resposta objetiva: ainda não temos estratégia. Colocar o gate de regime antes da
seleção de trades ajudou parcialmente, mas não resolveu a instabilidade.

Fast115 testou veto/gate focado no melhor `top 5%` e falhou forte. Fast116
mudou a arquitetura:

1. treina o modelo de payoff no treino;
2. gera predições out-of-fold nas janelas de treino;
3. escolhe um gate de regime usando só o treino OOF;
4. aplica o gate no holdout;
5. só depois seleciona o top fraction.

Ou seja: é um teste de regime antes da entrada, não um veto simples depois da
seleção.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Configuração

Runner:

`backend/scripts/research/run_book_regime_gate_payoff_scout.py`

Janelas:

- fast16, fast20, fast21, fast22, fast24, fast28, fast29, fast32, fast35,
  fast36, fast82, fast83, fast89.

Parâmetros:

- modelo: `hist_gradient_payoff`;
- horizonte: `300s`;
- target / breakeven activation: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `0.05`;
- gate keep fractions: `0.50`, `0.70`, `0.85`, `1.00`;
- latency: `100ms`;
- custo profit: `1 bps`;
- custo stress: `2 bps`;
- mínimo de janelas: `10`;
- mínimo operacional por janela: `3` execuções.

## Resultado

Status:

`REGIME_GATE_PAYOFF_FAILED_REPLICATION`

Motivo:

- `no_one_position_gate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

Resumo one-position:

| Métrica | Valor |
|---|---:|
| Execuções | `52` |
| Gross total | `+43.03297770737817` bps |
| Net total @ 1 bps | `-8.967022292621827` bps |
| Net total @ 2 bps | `-60.96702229262183` bps |
| Janelas positivas @ 1 bps | `6/13` |
| Janelas positivas @ 2 bps | `6/13` |
| Janelas com mínimo de 3 execuções | `11/13` |
| Pior janela @ 1 bps | `-6.425066433359014` bps |
| Exit reasons | `1 BREAKEVEN_STOP`, `7 INITIAL_STOP`, `42 TIMEOUT`, `2 TRAILING_STOP` |

Resumo independente:

| Métrica | Valor |
|---|---:|
| Execuções | `364` |
| Net total @ 1 bps | `+450.6774910874384` bps |
| Net total @ 2 bps | `+86.6774910874384` bps |
| Janelas positivas @ 1 bps | `8/13` |
| Janelas positivas @ 2 bps | `6/13` |
| Janelas com mínimo de 3 execuções | `13/13` |

## Por janela no modo one-position

| Janela | Execuções | Net @1bps | Net @2bps |
|---|---:|---:|---:|
| fast16 | `1` | `+4.343264375832244` | `+3.343264375832244` |
| fast20 | `4` | `+11.360253057055663` | `+7.3602530570556635` |
| fast21 | `5` | `+37.859297633896006` | `+32.859297633896006` |
| fast22 | `4` | `-14.146634040600864` | `-18.146634040600865` |
| fast24 | `3` | `-19.27519930007704` | `-22.27519930007704` |
| fast28 | `4` | `+13.43705201107624` | `+9.43705201107624` |
| fast29 | `4` | `+19.237297396809304` | `+15.237297396809304` |
| fast32 | `2` | `+8.426891211988943` | `+6.426891211988943` |
| fast35 | `4` | `-10.053694581883258` | `-14.053694581883258` |
| fast36 | `8` | `-45.99005256790503` | `-53.99005256790503` |
| fast82 | `5` | `-6.631335849821323` | `-11.631335849821323` |
| fast83 | `4` | `-1.3587007986063808` | `-5.358700798606381` |
| fast89 | `4` | `-6.175460840386329` | `-10.175460840386329` |

## Leitura

O regime-first melhorou uma parte do problema:

- no Fast114, `fast36` tinha `-113.81595768577692` bps @ 1 bps;
- no Fast116, `fast36` caiu para `-45.99005256790503` bps @ 1 bps.

Mas a melhoria não basta:

- o total one-position ficou negativo;
- só `6/13` janelas ficaram positivas com custo de `1 bps`;
- as perdas ficaram distribuídas em fast22, fast24, fast35, fast36, fast82,
  fast83 e fast89;
- o gate escolhido no treino ainda não generaliza de forma estável.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast116 mostra que um gate de regime antes da entrada é uma direção melhor que
gate pós-top-p, mas o formato atual — single-feature threshold + top 5% no mesmo
modelo de payoff — ainda não produz uma estratégia defensável.

O próximo avanço real teria que ser mais estrutural do que só mudar thresholds:

1. modelo de regime multivariado, validado antes do modelo de entrada;
2. mais janelas de book completo para separar regimes ruins sem overfit;
3. comparação multi-mercado para achar microestrutura com melhor estabilidade;
4. ou timeframe maior fora dos WALs curtos atuais.

## Comando executado

```bash
FAST116_WALS='fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m,fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m,fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m,fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m'

PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_regime_gate_payoff_scout.py \
  --wal-roots "$FAST116_WALS" \
  --output-root backend/data/microstructure_v1/reports/fast116-large-move-regime-first-top5-13w \
  --model hist_gradient_payoff \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.05 \
  --gate-keep-fractions 0.50,0.70,0.85,1.00 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 10 \
  --minimum-executable-per-window 3
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast116-large-move-regime-first-top5-13w/report.json`;
- report sha256:
  `715d2399b0f7f90dec0cbe39cb3e8859d8740a31888d41e012d181bf9fd8d3cb`.
