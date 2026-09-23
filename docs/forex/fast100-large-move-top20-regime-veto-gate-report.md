# Fast100 — gate causal em operações maiores com top 20%

Resposta objetiva: abrir o top-p para ganhar amostra piorou o resultado.

Fast99 mostrou que o candidato maior `target 10 / trailing 15 / top 5%` era
promissor, mas raro demais. Fast100 testou a hipótese natural: aumentar o top-p
para `20%` para dar mais material ao gate.

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

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --output-root backend/data/microstructure_v1/reports/fast100-large-move-top20-regime-veto-gate-audit \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.20 \
  --minimum-train-trades 20 \
  --minimum-train-labels 7 \
  --minimum-holdout-trades 3 \
  --maximum-train-negative-labels 0
```

## Status

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Motivos:

- `profit_positive_holdouts:3/10`;
- `stress_positive_holdouts:2/10`;
- `profit_negative_holdouts:1`;
- `no_strict_training_survivors:6`;
- `do_not_promote_to_strategy`.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| Holdouts | `10` |
| Training survivors | `4` |
| Sem survivor no treino | `6` |
| Profit-positive @ 1 bps | `3/10` |
| Stress-positive @ 2 bps | `2/10` |
| Profit-negative | `1` |
| Holdout trades avaliados | `17` |
| Holdout net total @ 1 bps | `+16.51705020996148` bps |
| Holdout net total @ 2 bps | `-0.4829497900385218` bps |
| Pior holdout @ 1 bps | `-15.162166706246486` bps |

## Por janela

| Holdout | Status | Execuções | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|
| fast16 | stress-fragile | `3` | `+0.4818419078622642` | `-2.518158092137736` |
| fast20 | sem survivor | `0` | `0` | `0` |
| fast21 | sem survivor | `0` | `0` | `0` |
| fast22 | stress-positive | `5` | `+8.953693207202466` | `+3.953693207202466` |
| fast24 | negativo | `5` | `-15.162166706246486` | `-20.162166706246488` |
| fast28 | sem survivor | `0` | `0` | `0` |
| fast29 | stress-positive | `4` | `+22.243681801143236` | `+18.243681801143236` |
| fast32 | sem survivor | `0` | `0` | `0` |
| fast35 | sem survivor | `0` | `0` | `0` |
| fast36 | sem survivor | `0` | `0` | `0` |

## Leitura

Fast100 mata a hipótese simples de "abre o top-p para ganhar amostra".

O sinal de operações maiores existe, mas parece concentrado nos scores mais
altos. Quando o top-p abre para `20%`, entram sinais piores, o gate não consegue
limpar e a replicação cai de `7/10` no Fast99 para `3/10`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O ramo de operações maiores fica classificado como:

- mais promissor que micro-alvos de `3 bps`;
- ainda insuficiente para bot;
- dependente de mais janelas/dados ou de uma arquitetura de seleção mais forte
  que não simplesmente aumente o top-p.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast100-large-move-top20-regime-veto-gate-audit/report.json`;
- report sha256:
  `51bdffe8bfefadc2657917f8414596e40b298f356d82054fb4a0b5f7663a3464`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast100-large-move-top20-regime-veto-gate-audit/holdout_results.jsonl`;
- holdout results sha256:
  `c54e7a00cd9e2445f1f8fececa844b4c88ef91328117114579d3a35a489df223`.
