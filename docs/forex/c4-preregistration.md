# Pré-registro C4: as duas estratégias publicadas de média positiva em C3, nos 31 mercados intocados

Registrado em 2026-09-24, depois do relatório de descoberta de C3 (`docs/forex/c3-discovery-report.md`) e **antes** de qualquer estratégia rodar no grupo B de C3.

## Por que existe e o que já se sabe (divulgação)

Em C3 nenhuma das 14 estratégias publicadas passou no critério nos 10 pares. Duas tiveram média positiva depois do custo, sem significância: **RSI2_D1** (+0,050 ATR por operação, t = 1,27, 64% de acerto, 1.546 operações) e **TSMOM** (+0,147 ATR, t = 1,02, 35% de acerto, 1.085 operações). Elas são escolhidas **por terem sido as melhores em C3**, ou seja, depois de ver o resultado. Por isso este teste é só nelas duas (K = 2) e só em dado que nenhuma estratégia deste projeto viu: os 31 mercados do grupo B de C3 (cruzados de FX, ouro, prata, índices de ações, petróleo, índice do dólar), 2012-01 a 2026-08, com o manifesto de dados já commitado (`docs/forex/c3-data-manifest.csv`).

## Regras (fixas)

- Código, custos, execução, unidade (ATR) e parâmetros exatamente os de C3 (`backend/scripts/research/fx_c3_lab.py`, commit deste documento); nada muda.
- **Aprovada** se todas: média > 0 no custo base e no estresse; valor-p unilateral do t por cluster de dia FX ≤ 0,05 com correção de Holm entre as 2; ao menos 55% dos instrumentos com média > 0; ao menos 200 operações.
- O resultado é registrado em `docs/forex/c4-report.md` e em `docs/forex/c4-registry.jsonl`, com o commit do código.
- Se aprovada: próximo passo é a confirmação prospectiva (dados com entrada depois deste commit, nos 41 mercados), não dinheiro. Se não: paro e levo os números ao Igor.

## Palpite antes do dado

TSMOM tem a evidência publicada mais forte justamente fora do FX (índices, commodities); RSI2 foi publicada para ações. Palpite: 25% a 35% de chance de pelo menos uma passar.
