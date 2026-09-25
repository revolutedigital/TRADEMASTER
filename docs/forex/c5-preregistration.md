# Pré-registro C5: TSMOM e RSI2 em índices de ações, em períodos e mercados que ninguém do projeto testou

Registrado em 2026-09-24, depois dos relatórios de C3 e C4 e **antes** de baixar os dados deste teste e de rodar qualquer estratégia neles.

## De onde vem a hipótese (divulgação)

Em C4 nenhuma das duas estratégias foi aprovada nos 31 mercados novos, mas a quebra por mercado mostrou um padrão que a literatura já descrevia: **TSMOM** (seguir tendência de 12 meses) foi positiva em todos os índices de ações, no ouro, na prata, no petróleo e no índice do dólar, e negativa em quase todos os cruzados de FX; **RSI2** (Larry Connors, publicada para ações) teve os melhores resultados nos índices dos EUA. A hipótese nasceu olhando esse dado, então ela **não pode** ser confirmada nele: este teste só usa dado que nenhuma estratégia do projeto viu.

## Hipótese

Em índices de ações à vista, TSMOM e RSI2_D1, com as regras e parâmetros de C3 sem nenhuma mudança, têm resultado médio positivo depois de custo e de financiamento.

## Dados (baixados depois deste commit)

Fonte: Yahoo Finance, barras diárias (abertura, máxima, mínima, fechamento) dos índices à vista. Índices à vista não têm rolagem de contrato, o que evita o viés dos contratos futuros contínuos sem ajuste (por isso ouro, petróleo e outras commodities ficam de fora, apesar do resultado de C4).

- **Amostra X, período intocado dos índices já vistos:** S&P 500 (^GSPC), Nasdaq 100 (^NDX), DAX (^GDAXI), Nikkei 225 (^N225), CAC 40 (^FCHI), S&P/ASX 200 (^AXJO), Hang Seng (^HSI), FTSE 100 (^FTSE), de 2000-01-01 a 2011-12-31 (C3 e C4 usaram esses mercados só de 2012 em diante).
- **Amostra Y, índices que o projeto nunca usou:** Russell 2000 (^RUT), IBEX 35 (^IBEX), SMI (^SSMI), AEX (^AEX), S&P/TSX (^GSPTSE), KOSPI (^KS11), TAIEX (^TWII), Ibovespa (^BVSP), IPC México (^MXX), Straits Times (^STI), de 2000-01-01 a 2026-08-31.
- **Qualidade (só de dado, antes de rodar):** um dia entra se abertura, máxima, mínima e fechamento forem positivos e a máxima ≥ mínima; um índice entra se tiver ao menos 5 anos de dias válidos na sua amostra e se no máximo 20% dos dias tiverem a abertura igual ao fechamento anterior (sinal de abertura que a fonte não registra). Índice reprovado sai da amostra antes de qualquer estratégia rodar, com o motivo no relatório.

## Execução e custos

- Sinal no fechamento do dia; execução na abertura do dia seguinte; saídas por regra também na abertura seguinte. Sem stop (nenhuma das duas estratégias tem stop no original).
- Custo por operação completa: 0,05% do preço no base e 0,10% no estresse (spread e slippage de CFD de índice).
- **Financiamento:** 3% ao ano do valor da posição por dia corrido em posição no base, 6% no estresse, comprado ou vendido (aproximação conservadora do swap de CFD de índice).
- Resultado em ATR(20) diário no fechamento do dia do sinal, como em C3.

## Critério (K = 2)

Uma estratégia é **aprovada** se, **separadamente em X e em Y**: média > 0 no base e no estresse; valor-p unilateral do t por cluster de dia de entrada ≤ 0,05 com correção de Holm entre as 2 estratégias; ao menos 55% dos índices com média > 0 no base; ao menos 150 operações.

Se aprovada: confirmação prospectiva (entradas depois deste commit) em demo, nos índices que a Fusion oferece, antes de qualquer dinheiro. Se nada for aprovado: paro e levo os números ao Igor.

## Palpite antes do dado

TSMOM em índices tem décadas de evidência publicada, mas com poucas operações por índice e muito ruído; RSI2 perdeu força depois de publicada (1990-2000). Palpite: 30% a 40% de chance de TSMOM passar nas duas amostras, 15% a 25% para RSI2.
