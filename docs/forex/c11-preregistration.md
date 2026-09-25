# Pré-registro C11: day trade em fluxo de dinheiro e calendário

Registrado em 2026-09-25, antes de qualquer resultado desta rodada. Nenhuma média, taxa de acerto ou t destas janelas foi calculada. Só foram contadas as entradas por mercado, para conferir que os dados cobrem as janelas. Código: `backend/scripts/research/fx_c11_lab.py`, commitado junto.

## Por que esta rodada

C10 mostrou que regras técnicas de day trade publicadas não têm vantagem depois do custo, com nenhum formato de stop ou alvo: 4.464 configurações e o melhor t foi 1,0. Esta rodada troca de categoria. Testa horários em que existe um **fluxo de dinheiro conhecido e documentado**, com razão econômica para o preço andar numa direção.

## As 11 janelas (fixas, todas zeradas antes do fim do dia FX, sem swap)

| Janela | Mercados | Lado | Horário | Fonte |
|---|---|---|---|---|
| Antes do fix de Tóquio | 7 pares com USD + índice do dólar | compra USD | 08:55–09:55 Tóquio | Krohn, Mueller e Whelan (Journal of Finance, 2024): o dólar sobe antes dos fixes e cai depois |
| Depois do fix de Tóquio | idem | vende USD | 09:55–10:55 Tóquio | idem |
| Antes do fix do BCE | idem | compra USD | 13:15–14:15 Frankfurt | idem |
| Depois do fix do BCE | idem | vende USD | 14:15–15:15 Frankfurt | idem |
| Antes do fix de Londres (WM/R) | idem | compra USD | 15:00–16:00 Londres | idem |
| Depois do fix de Londres | idem | vende USD | 16:00–17:00 Londres | idem |
| Fim de mês, antes do fix de Londres | idem | vende USD se a bolsa americana ganhou das estrangeiras no mês até a véspera, compra se perdeu | 15:00–16:00 Londres, último dia útil do mês | Melvin e Prins (Journal of Financial Markets, 2015): rebalanceamento do hedge cambial de quem tem ações |
| Fim de mês, depois do fix | idem | o contrário (reversão) | 16:00–17:00 Londres | idem |
| Deriva noturna dos EUA | S&P 500, Nasdaq 100 | compra | 18:05–09:25 Nova York | Boyarchenko, Larsen e Whelan (Review of Financial Studies, 2023) |
| Abertura europeia | S&P 500, Nasdaq 100 | compra | 02:00–04:00 Nova York | idem (a alta se concentra na abertura da Europa) |
| Virada do mês | os 8 índices de ações | compra | o dia FX inteiro (do 1º minuto a partir das 18:05 Nova York até o fim), no último dia útil do mês e nos 3 primeiros | Lakonishok e Smidt (1988); McConnell e Xu (2008) |

O sinal do fim de mês usa fechamentos diários do Yahoo (S&P 500 contra a média de DAX, FTSE, Nikkei, CAC, ASX 200 e Hang Seng), do último fechamento do mês anterior ao último antes do dia. Dias úteis são de segunda a sexta, sem calendário de feriados.

Uma janela da deriva noturna europeia foi desenhada e **retirada antes de qualquer resultado**: nesta fonte, DAX, CAC e FTSE não têm dado de madrugada até cerca de 2020.

## Execução e custo

Entrada no primeiro minuto até 5 minutos depois do início da janela (a virada do mês aceita até 10 horas, para esperar a abertura de cada índice). Saída no fechamento do último minuto da janela, ou no fim do dia FX. Stop de proteção de 1 ATR diário, checado minuto a minuto, primeiro dentro do minuto. Custos de C3; estresse com custo dobrado. Unidade: R = 1 ATR diário.

## Julgamento

**Descoberta, 2012-01 a 2026-08 (manifesto de C3):** média de R por operação e t agrupado por dia FX. Bootstrap de máximo t sobre as 11 janelas (5.000 sorteios, bloco médio de 10 dias, semente 20261011). **Passa:** p ajustado ≤ 0,05; média > 0 com custo normal e dobrado; ≥ 55% dos mercados (com 20+ operações) positivos; ≥ 200 operações.

**Período intocado, 2003-01 a 2011-12 (manifesto de C10, já commitado):** só as que passarem. Uma rodada. p unilateral pelo t agrupado, Holm entre elas. **Aprovada:** p de Holm ≤ 0,05; média > 0 com custo normal e dobrado; ≥ 300 operações; ≥ 55% dos mercados positivos. Os índices só têm 2010–2011 nesse período.

**Placebo:** cada aprovada roda em 20 cópias da descoberta com o sinal de cada minuto trocado ao acaso. Só fica aprovada se no máximo 1 cópia alcançar a média real.

Aprovada vai para a demo 10139135 (0,5% do capital por ATR diário, ou o volume mínimo). Emendas só antes de resultado, em `c11-registry.jsonl`.
