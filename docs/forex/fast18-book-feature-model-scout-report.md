# Rodada 18 — modelo leak-safe com features de book

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast16 de 18 minutos  
Resultado: sem estratégia; scout corrigiu vazamento e não encontrou separação confiável

## Pergunta

Depois da rodada 17, a dúvida era se as features de book/flow conseguiam escolher entradas de
0,5–3 bps melhor que o acaso. A primeira exploração local pareceu mostrar um sinal forte, mas o motivo
era vazamento: `mfe_bps`, `mae_bps` e `terminal_bps` estavam entrando como features. Essas colunas são
futuro do caminho, portanto trapaceiam.

Esta rodada transforma essa checagem em script auditável com guarda explícita contra vazamento.

## O que foi implementado

Script novo:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_feature_model_scout.py --wal-root <wal> --output-root <out> --horizons-seconds 30 --thresholds-bps 0.5,1,2,3 --top-fractions 0.05,0.10,0.20 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 10`

O script:

- materializa rows com `trade + depth + mark + spot`;
- recalcula labels por alvo;
- exclui colunas futuras e de label antes do treino;
- treina `logistic_balanced` e `hist_gradient`;
- usa blocos temporais: treina nos blocos anteriores e testa no bloco seguinte;
- mede top 5%, 10% e 20%;
- reporta payoff bruto do caminho apenas como diagnóstico;
- mantém `approved_strategy=false`, `order_submission_allowed=false` e `execution_authorization=none`.

Guarda de vazamento:

| Coluna | Status |
|---|---|
| `mfe_bps` | excluída |
| `mae_bps` | excluída |
| `terminal_bps` | excluída |
| timestamps de toque/stop/target | excluídos |
| target e flags de pagamento | excluídos |

## Resultado em 30s

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_feature_model_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-feature-model-scout-fast18-18m --horizons-seconds 30 --thresholds-bps 0.5,1,2,3 --top-fractions 0.05,0.10,0.20 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 10`

Resumo:

| Campo | Valor |
|---|---:|
| Decisões | 204 |
| Features causais | 122 |
| Folds de teste | 2 |
| Estratégia aprovada | não |

Melhor candidato por payoff bruto:

| Campo | Valor |
|---|---:|
| Modelo | `logistic_balanced` |
| Alvo | 3 bps |
| Top-p | 5% |
| Selecionados | 12 |
| Target rate base | 4,90% |
| Target rate selecionado | 0,00% |
| Payoff bruto base | +0,033 bps |
| Payoff bruto selecionado | +0,408 bps |

Leitura: o melhor candidato positivo não acertou o alvo nenhuma vez. O ganho bruto veio de timeouts
favoráveis pequenos, não de prever o toque de 3 bps. Isso não é edge operacional.

Outro candidato aparentemente melhor para alvo de 1 bps:

| Campo | Valor |
|---|---:|
| Modelo | `logistic_balanced` |
| Alvo | 1 bps |
| Top-p | 5% |
| Selecionados | 12 |
| Target rate base | 27,94% |
| Target rate selecionado | 25,00% |
| Payoff bruto selecionado | +0,245 bps |

Também não serve: a taxa de alvo selecionada ficou abaixo da base.

## Decisão

Ainda não temos estratégia.

A rodada 18 fez duas coisas importantes:

1. impediu uma falsa descoberta causada por vazamento de futuro;
2. mostrou que, sem vazamento, o modelo não separou entradas de 30s com evidência suficiente.

Payoffs brutos abaixo de 1 bp não sobrevivem a custo/slippage normal sem uma vantagem de execução
muito específica. Esta rodada não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Script: `backend/scripts/research/run_book_feature_model_scout.py`
- Teste: `backend/tests/unit/test_book_feature_model_scout.py`
- Relatório Fast18 ignorado pelo Git:
  `backend/data/microstructure_v1/reports/book-feature-model-scout-fast18-18m/report.json`
  (`3b95cac7ae275255266795db676be26e7e100225dca52aff073e0ed2f854562d`)
