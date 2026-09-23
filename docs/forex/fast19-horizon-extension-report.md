# Rodada 19 — horizonte 60s/120s no book prospectivo

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast16 de 18 minutos  
Resultado: horizonte maior revela drift bruto no recorte, mas não uma estratégia fora da amostra

## Pergunta

As rodadas 16 a 18 rejeitaram a versão curta de 30s. A próxima hipótese era que o movimento existe,
mas precisa de mais tempo para aparecer. Esta rodada repete a sensibilidade e o modelo leak-safe em
60s e 120s.

## Sensibilidade — 60s

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_threshold_sensitivity.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast19-60s --horizons-seconds 60 --thresholds-bps 0.5,1,2,3,5,8,12,24 --decision-stride-seconds 5 --minimum-decisions-for-modeling 180`

Resumo:

| Campo | Valor |
|---|---:|
| Decisões | 198 |
| Melhor lado bruto | SELL |
| Melhor alvo bruto com toque | 1 bps |
| Toque SELL 1 bps | 60,61% |
| Payoff bruto SELL 1 bps | +0,197 bps |
| Terminal médio SELL | +0,154 bps |

Em 60s apareceu um drift bruto vendido no recorte. Mas ele é pequeno demais para ser considerado
edge líquido sem custo muito baixo.

## Modelo leak-safe — 60s

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_feature_model_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-feature-model-scout-fast19-60s --horizons-seconds 60 --thresholds-bps 0.5,1,2,3,5 --top-fractions 0.05,0.10,0.20 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 10`

Melhor candidato:

| Campo | Valor |
|---|---:|
| Modelo | `logistic_balanced` |
| Alvo | 0,5 bps |
| Top-p | 5% |
| Selecionados | 10 |
| Target rate base | 48,98% |
| Target rate selecionado | 30,00% |
| Payoff bruto base | -0,532 bps |
| Payoff bruto selecionado | +0,328 bps |

Leitura: o payoff selecionado melhorou em relação ao teste, mas a taxa de toque caiu muito. Não é uma
prova de entrada; parece mais captura de micro-drift/timeout no recorte.

## Sensibilidade — 120s

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_threshold_sensitivity.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast19-120s --horizons-seconds 120 --thresholds-bps 0.5,1,2,3,5,8,12,24 --decision-stride-seconds 5 --minimum-decisions-for-modeling 170`

Resumo:

| Campo | Valor |
|---|---:|
| Decisões | 186 |
| Melhor lado bruto | SELL |
| Terminal médio SELL | +0,472 bps |
| Toque SELL 1 bps | 73,66% |
| Payoff bruto SELL 5 bps | +0,221 bps |
| Payoff bruto SELL 8 bps | +0,468 bps |

Esse resultado parece bonito, mas é enganoso se lido como estratégia: grande parte vem de ficar
vendido durante um trecho em que o mercado derivou para baixo. É uma constatação do recorte, não uma
regra causal.

## Modelo leak-safe — 120s

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_feature_model_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-feature-model-scout-fast19-120s --horizons-seconds 120 --thresholds-bps 0.5,1,2,3,5 --top-fractions 0.05,0.10,0.20 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 10`

Melhor candidato:

| Campo | Valor |
|---|---:|
| Modelo | `logistic_balanced` |
| Alvo | 0,5 bps |
| Top-p | 5% |
| Selecionados | 10 |
| Target rate base | 59,24% |
| Target rate selecionado | 60,00% |
| Payoff bruto base | -1,034 bps |
| Payoff bruto selecionado | -0,195 bps |

Em 120s, mesmo o melhor candidato fora da amostra ficou negativo.

## Decisão

Ainda não temos estratégia.

A rodada 19 mostra que aumentar o horizonte melhora a reachability, mas não resolveu o problema
central: ainda não há filtro causal fora da amostra que transforme os movimentos em payoff líquido
operável.

O próximo caminho real não é “apertar mais um top-p” nesse mesmo recorte. Precisamos de mais dados
prospectivos e de um desenho de confirmação por regime:

1. capturar várias janelas, incluindo abertura dos EUA/alta volatilidade;
2. repetir 60s/120s por janela e por regime;
3. só então congelar uma regra candidata e validar em captura futura;
4. manter Demo/Testnet/LIVE bloqueados até passar gate estatístico.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Relatório sensibilidade 60s:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast19-60s/report.json`
  (`fc464e5503566b463f3bd07b209a1a2f6699c3616b2410d3484c0cbc9e24aac7`)
- Relatório sensibilidade 120s:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast19-120s/report.json`
  (`15aecf806bee9aff2f47b7262413e006f3f9b93eae114aca9f6d091dcf651bf7`)
- Relatório modelo 60s:
  `backend/data/microstructure_v1/reports/book-feature-model-scout-fast19-60s/report.json`
  (`a88d1dce73680f2f07364fe8b2feabc2c5d3d1d2f428b39cd15b4c8a394dbf9e`)
- Relatório modelo 120s:
  `backend/data/microstructure_v1/reports/book-feature-model-scout-fast19-120s/report.json`
  (`18ce94bb4f11ce0a5959cd376edb331c36d59a2a30cab80c6baa92d06a13d284`)
