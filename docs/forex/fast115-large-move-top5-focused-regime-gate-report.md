# Fast115 — gate focado no large-move top 5% com 13 janelas

Resposta objetiva: ainda não temos estratégia. O teste que faltava no ramo
large-move ficou negativo.

Fast114 confirmou que, dentro da curva fina de top-p, o melhor ponto ainda era:

```text
hist_gradient_payoff
target 10 bps
trailing distance 15 bps
top 5%
```

Mas o candidato ainda quebrava em `fast36` e não replicava sob custo de stress.
Fast115 testou a próxima pergunta certa:

> Existe um gate causal, treinado só nas janelas de treino, que filtra os regimes
> ruins desse `top 5%` quando ampliamos para 13 janelas?

Resposta: não.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Configuração

Runner:

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Janelas:

- fast16, fast20, fast21, fast22, fast24, fast28, fast29, fast32, fast35,
  fast36, fast82, fast83, fast89.

Parâmetros:

- modelo: `hist_gradient_payoff`;
- horizonte: `300s`;
- target / breakeven activation: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `0.05`;
- gate features focadas por keywords:
  `return`, `spread`, `vol`, `volume`, `imbalance`;
- gate keep fractions: `0.50`, `0.70`, `0.85`, `1.00`;
- latency: `100ms`;
- mínimo de treino: `10` trades;
- mínimo de labels no treino: `9`;
- mínimo no holdout: `3` trades;
- máximo de labels negativas no treino: `0`;
- custo profit: `1 bps`;
- custo stress: `2 bps`.

## Resultado

Status:

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Resumo:

| Métrica | Valor |
|---|---:|
| Holdouts | `13` |
| Training survivors | `2` |
| Holdouts sem survivor estrito | `11` |
| Profit-positive holdouts | `0/13` |
| Stress-positive holdouts | `0/13` |
| Profit-negative holdouts | `2` |
| Holdout trades avaliados | `12` |
| Net total @ 1 bps | `-124.60488121310692` bps |
| Net total @ 2 bps | `-136.60488121310692` bps |
| Pior holdout @ 1 bps | `-113.81595768577692` bps |
| Pior holdout @ 2 bps | `-121.81595768577692` bps |

Holdouts sem survivor:

```text
fast16, fast20, fast21, fast22, fast24, fast28, fast29,
fast32, fast82, fast83, fast89
```

Holdouts negativos:

```text
fast35, fast36
```

## Leitura

Fast115 fecha a lacuna entre Fast99 e Fast107:

- Fast99: `top 5%` em 10 janelas parecia promissor, mas raro;
- Fast107: `top 20%` em 13 janelas morreu no treino;
- Fast114: curva fina mostrou que `top 5%` ainda era o melhor ponto;
- Fast115: `top 5%` em 13 janelas com gate focado também falhou.

O problema não é simplesmente escolher melhor o top-p nem adicionar um veto
causal simples pós-score. O modelo ainda não identifica de forma estável quando
o regime é ruim antes do trade.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Encerrar o ramo:

```text
large-move BTCUSDT book-payoff
target 10 bps
trail 15 bps
top fraction + gate causal simples
```

como não estratégia nos WALs atuais.

Se continuar, a próxima mudança precisa ser estrutural:

1. modelo de regime anterior à entrada, não gate pós-top-p;
2. mais janelas/dados de book completos;
3. multi-mercado/multi-par para comparar onde a microestrutura é mais limpa;
4. timeframe maior fora do mesmo conjunto de WALs curtos.

## Comando executado

```bash
FAST115_WALS='fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m,fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m,fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m,fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m'

PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --wal-roots "$FAST115_WALS" \
  --output-root backend/data/microstructure_v1/reports/fast115-large-move-top5-focused-regime-gate-13w \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.05 \
  --gate-feature-keywords return,spread,vol,volume,imbalance \
  --minimum-train-trades 10 \
  --minimum-train-labels 9 \
  --minimum-holdout-trades 3 \
  --maximum-train-negative-labels 0
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast115-large-move-top5-focused-regime-gate-13w/report.json`;
- report sha256:
  `edd7c6fc7a3352b9a7b2e7ba2162ab618ef8175aeeca94f1626e0ca07b690686`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast115-large-move-top5-focused-regime-gate-13w/holdout_results.jsonl`;
- holdout results sha256:
  `d099db74d1eeb79979c01396d881aad735ecaf9a8a82db28a9e44a9531bc70bd`.
