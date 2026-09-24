# Fast107 — top 20% no ramo large-move/gate focado

Resposta objetiva: não temos estratégia. Abrir o top fraction para `20%` não
salvou o ramo de operações maiores com gate de regime.

Fast99 tinha deixado uma pendência plausível: o candidato `target 10 bps`,
`trailing 15 bps`, `top 5%` era raro demais. A hipótese Fast107 foi abrir para
`top 20%` para dar mais amostra ao gate, mantendo validação leave-one-window-out
e exigindo zero janelas negativas no treino.

## Safety boundary

Mantido em todos os passos:

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

## Ajustes de engenharia antes do run

O runner de gate ficou caro quando o top fraction subiu para `20%`. Foram feitos
dois ajustes sem mudar a lógica estatística:

1. `_apply_one_position_constraint_by_score` deixou de converter cada linha para
   `dict` e passou a selecionar por índice;
2. `run_fast96_fast95_regime_gate_audit.py` ganhou caminho presorted para evitar
   sort/groupby repetido por gate;
3. foi adicionada a flag `--gate-feature-keywords` para rodar gates focados em
   famílias causais relevantes, preservando o default amplo.

Validação:

- `backend/tests/unit/test_book_multi_window_model_scout.py`;
- `backend/tests/unit/test_fast96_fast95_regime_gate_audit.py`;
- resultado: `12 passed`;
- ruff: `All checks passed!`.

## Configuração

Runner:

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Janelas:

- fast16, fast20, fast21, fast22, fast24, fast28, fast29, fast32, fast35,
  fast36, fast82, fast83, fast89.

Parâmetros:

- modelo: `hist_gradient_payoff`;
- horizonte: `300s`;
- target / trailing activation: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `0.20`;
- gate features focadas por keywords:
  `return, spread, vol, volume, imbalance`;
- gate keep fractions: `0.50`, `0.70`, `0.85`, `1.00`;
- latency: `100ms`;
- mínimo de treino: `20` trades;
- mínimo de labels no treino: `9`;
- mínimo no holdout: `3` trades;
- máximo de labels negativas no treino: `0`;
- custo profit: `1 bps`;
- custo stress: `2 bps`.

## Resultado

Status:

`FAST96_NO_STRICT_TRAINING_SURVIVORS`

Resumo:

| Métrica | Valor |
|---|---:|
| Holdouts | `13` |
| Training survivors | `0` |
| Holdouts sem survivor estrito | `13` |
| Holdout trades avaliados | `0` |
| Profit-positive holdouts | `0` |
| Stress-positive holdouts | `0` |
| Estratégia aprovada | `false` |

Todos os holdouts falharam antes da avaliação independente porque nenhum gate
foi bom o bastante no treino sob a regra de zero labels negativas.

## Detalhe por holdout

Cada holdout avaliou:

- `122` features de modelo;
- `66` gate features após filtro focado;
- `397` gates;
- cerca de `1319` a `1377` sinais selecionados no treino.

Resultado por holdout:

| Holdout | Status |
|---|---|
| fast16 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast20 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast21 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast22 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast24 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast28 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast29 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast32 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast35 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast36 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast82 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast83 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |
| fast89 | `FAST96_NO_STRICT_TRAINING_SURVIVOR` |

## Decisão

Encerrar esta pendência do ramo Fast99/Fast104 como não estratégia.

Abrir o top para `20%` aumentou amostra, mas não encontrou um gate estável o
suficiente. Esse resultado é pior que um holdout fraco: o modelo/gate não passa
sequer no treino leave-one-window-out com filtro focado.

## O que isso ensina

- O problema não era apenas falta de amostra no `top 5%`.
- O ramo large-move continua mais intuitivo que micro-scalp, mas não oferece
  estabilidade suficiente nos WALs atuais.
- Relaxar o critério de zero janelas negativas poderia gerar diagnóstico, mas
  não uma estratégia que eu defenderia para Testnet.
- Continuar tunando top fraction/gate nesse mesmo conjunto tende a overfit.

## Próximo caminho recomendado

Não continuar no ramo `target 10 / trailing 15 / top fraction + gate` sobre os
mesmos WALs.

Se a busca continuar, a próxima mudança precisa ser estrutural:

1. baixar dados multi-par e comparar mercados;
2. ou usar book/depth longo, não apenas WALs curtos;
3. ou fazer modelo de regime anterior à entrada, não gate pós-top-p;
4. ou formular estratégia de timeframe maior fora de microestrutura BTCUSDT.

## Comando executado

```bash
FAST107_WALS='fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m,fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m,fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m,fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m'

PYTHONPATH=backend ./backend/.venv/bin/python backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --wal-roots "$FAST107_WALS" \
  --output-root backend/data/microstructure_v1/reports/fast107-large-move-top20-focused-regime-gate-audit \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.20 \
  --gate-feature-keywords return,spread,vol,volume,imbalance \
  --minimum-train-trades 20 \
  --minimum-train-labels 9 \
  --minimum-holdout-trades 3 \
  --maximum-train-negative-labels 0
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast107-large-move-top20-focused-regime-gate-audit/report.json`;
- report sha256:
  `c54311bb4ed479fadea0067a9b8869f58d3fd75e5d301be77f05f2716532392e`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast107-large-move-top20-focused-regime-gate-audit/holdout_results.jsonl`;
- holdout results sha256:
  `2a3240bf37e9a24e7f1884a441ecc03a4e4d3592e981eaacb6e7a628dfd13780`.
