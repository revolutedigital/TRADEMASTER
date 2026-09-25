# Pré-registro C10: day trade com gestão assimétrica, em todos os mercados

Registrado em 2026-09-25, antes de qualquer resultado da busca. O código (`fx_c10_core.py`, `fx_c10_families/`, `fx_c10_lab.py`) é commitado junto com este texto. O rodador recusa rodar com o código ou este arquivo fora do commit.

## Pergunta

Existe regra de day trade publicada que, com stop e alvo assimétricos (realização parcial, stop no zero a zero, runner, trailing), dê resultado positivo por operação, líquido de custo, nos 37 mercados, e que se repita num período que ninguém tocou?

## O que entra

**Famílias e parâmetros** (34 conjuntos, fixos; fontes nos módulos):

| Família | Barras | Parâmetros | Stop estrutural |
|---|---|---|---|
| Rompimento da faixa de abertura (Crabel 1990) | M5 | faixa de 15, 30 ou 60 min; 30 min só a favor da tendência diária | outro lado da faixa |
| Primeiro candle de 5 min (Zarattini, Barbon e Aziz 2024) | M5 | — | outro extremo do candle |
| Momentum intradiário (Gao et al. 2018; Elaut et al. 2018) | M5 | só a 1ª meia hora; 1ª e penúltima meia hora juntas | não |
| RSI(2) de Connors | M15, M30 | limiares 5/95 e 10/90, com média de 200 barras | não |
| Bandas de Bollinger (reversão) | M15, M30 | 20 barras, 2 e 2,5 desvios | não |
| Donchian intradiário com filtro de tendência diária | M15, H1 | 20 e 55 barras | canal oposto de N/2 |
| Rompimento de volatilidade de Larry Williams | M5 | k = 0,3, 0,5, 0,7 da faixa de ontem | abertura do dia |
| Gap de abertura (reversão) | M5 | gap acima de 0,25 e 0,5 ATR diário | não |
| NR4 e NR7 (Crabel) | M5 | — | outro extremo do dia estreito |
| Explosão de momentum: seguir e reverter | M5 | barra de 2,5 e 4 vezes a média de 20 | extremo da barra |
| Rompimento da faixa asiática (00:00–07:00 Londres) | M5 | — | outro lado da faixa |
| Rompimento da máxima/mínima de ontem | M5 | — | extremo oposto do dia até ali |
| Turtle Soup (Raschke e Connors 1995) | M15 | 20 dias (extremo com 4+ dias) e 1 dia | extremo do dia até ali |

**Sessões:** índices usam o pregão à vista da bolsa principal (Nova York 09:30–16:00, dólar 08:20–16:00, Frankfurt e Paris 09:00–17:30, Londres 08:00–16:30, Tóquio 09:00–15:00, Hong Kong 09:30–16:00, Sydney 10:00–16:00); pares de moedas usam Londres 08:00–16:30.

**Gestão (144 linhas, a mesma para todas as famílias):** stop inicial {estrutural da família, 0,25, 0,5, 1 ATR diário} × 12 formatos de saída {alvo único de 0,5, 1, 2 ou 3 R; metade em 0,5 ou 1 R e o resto em 3 ou 6 R; metade em 0,5 ou 1 R e o resto com trailing de 0,5 ou 1 R} × tempo máximo {60 min, 240 min, até o fim do dia}. Depois da primeira realização, o stop do resto vai para o preço de entrada. Famílias sem stop estrutural não rodam as 36 linhas de stop estrutural.

**Execução (igual para todas):** sinal no fechamento da barra; entrada na abertura do minuto seguinte. Uma posição por vez em cada mercado e configuração. Tudo zerado no fim do dia FX (17:00 de Nova York), e nenhuma entrada com menos de 15 minutos para o fim. Dentro do minuto, o stop é checado primeiro. Gap além do stop executa na abertura. O runner nunca executa no mesmo minuto da primeira realização. O stop estrutural nunca fica a menos de 0,1 ATR diário (se ficar, é alargado até lá).

**Custos:** os de C3 por mercado, ida e volta; estresse com custo dobrado. **Unidade:** R, a distância do stop inicial, líquida de custo.

**Mercados:** os 37 da banca (C9: sem ouro, prata, WTI e Brent).

## Descoberta: 2012-01 a 2026-08 (manifesto de C3)

Esse período já foi usado pelas rodadas diárias e de H1 (C3 a C9) e, em parte, pelo lab rápido de 10 pares. As regras de day trade acima, com esta gestão, não foram rodadas nele. Por isso a prova de verdade é o período intocado.

- Estatística de cada configuração: média de R por operação e t com erro agrupado pelo dia FX da entrada.
- Correção pela busca inteira: bootstrap estacionário de máximo t (5.000 sorteios, bloco médio de 10 dias, semente 20261001) sobre todas as configurações que tiveram operações.
- **Etapa A:** p ajustado ≤ 0,05; média > 0 com custo normal e com custo dobrado; ≥ 55% dos mercados (com 20+ operações) com média positiva; ≥ 500 operações.

**Levadas ao período intocado (no máximo 10):** todas que passarem na etapa A, pela ordem de t. Se forem menos de 10, completam a lista as melhores (por t) de cada outro conjunto família-parâmetro que passarem em tudo da etapa A menos o p, uma por conjunto.

## Período intocado: 2003-01 a 2011-12

Dados M1 da HistData baixados em 2026-09-25. Nenhum resultado foi calculado neles. O manifesto segue as regras de C3 (mês com 60%+ dos minutos da mediana, mercado com 36+ meses bons) e é gerado e commitado antes da rodada; mercados sem dado no período não entram.

- Cada candidata roda uma vez. p unilateral pelo t agrupado; **Holm** sobre as candidatas levadas.
- **Aprovada:** p de Holm ≤ 0,05; média > 0 com custo normal e dobrado; ≥ 300 operações; ≥ 55% dos mercados positivos.
- **Placebo:** cada aprovada roda também em 20 cópias do período de descoberta com o sinal de cada minuto trocado ao acaso (sem vantagem possível). Só fica aprovada se no máximo 1 cópia alcançar a média real da descoberta, o que pega qualquer vazamento de informação do futuro no código.

## Depois

Aprovada vai para a demo 10139135 com estas mesmas regras, 0,5% do capital em risco por operação (1 R), ou o volume mínimo. Sem aprovada, o relatório mostra as melhores com taxa de acerto, operações por dia, ganho e perda médios, e nada vai para a demo como se tivesse vantagem.

Emendas só antes de resultado, registradas em `c10-registry.jsonl`.
