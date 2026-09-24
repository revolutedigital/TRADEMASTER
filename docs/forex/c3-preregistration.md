# Pré-registro C3: estratégias publicadas, com os parâmetros de quem as publicou, em 41 mercados

Registrado em 2026-09-24, **antes** de existir o código destas estratégias e antes de qualquer resultado delas em qualquer dado. Muda só por emenda registrada em `docs/forex/c3-registry.jsonl` (hash deste documento e de `docs/forex/c3-configs.json`), nunca por causa de resultado.

## Por que existe

O Igor quer entradas frequentes em vários mercados, stop pequeno perto do ganho possível e o vencedor carregado. As rodadas anteriores (rodada 1, rodada 2 e as ~180 rodadas `fastN` do Codex) testaram sobretudo regras de curto prazo inventadas ou ajustadas pelo próprio projeto, em 10 pares de FX e em cripto, e nenhuma passou depois do custo. Esta rodada muda três coisas:

1. **Estratégias copiadas de quem as publicou, com os parâmetros originais.** Nenhum parâmetro é escolhido por nós; não existe grade. Isso zera o grau de liberdade de ajuste.
2. **Mercados que nenhuma estratégia deste projeto viu:** 31 instrumentos novos (cruzados de FX, ouro, prata, índices de ações, petróleo, índice do dólar).
3. **Prova fora do mercado:** o que for aprovado nos 10 pares já conhecidos precisa funcionar, sem mudar nada, nos 31 mercados novos; depois, uma única vez, no dado que ainda não existe hoje.

## Divulgação

Os 10 pares (grupo A) já foram muito minerados por este projeto de 2011 a 2026, inclusive regras parecidas com duas destas famílias (o G0 testou `sma_rsi` e `bollinger_reversion` em D1; o Codex testou canal Donchian D1 em sentido contrário, como reversão). Por isso o grupo A serve só de descoberta, e a prova real é o grupo B. Quem escreve este documento conhece a literatura de onde as estratégias vêm e, em termos gerais, o comportamento dos mercados nesses anos.

## Dados

- **Fonte:** HistData, M1 em ASCII (só bid), 2012-01 a 2026-08, relógio EST fixo (UTC−5 sem horário de verão, segundo a documentação da fonte; conferido contra o M1 já validado do projeto antes de qualquer estratégia rodar, e a leitura muda se a conferência mostrar outra coisa, registrado como emenda).
- **Grupo A, descoberta (10):** EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, GBPJPY, EURGBP.
- **Grupo B, réplica (31):** EURCHF, AUDJPY, CADJPY, CHFJPY, NZDJPY, EURAUD, EURCAD, EURNZD, GBPAUD, GBPCAD, GBPCHF, GBPNZD, AUDCAD, AUDCHF, AUDNZD, NZDCAD, NZDCHF, CADCHF, XAUUSD, XAGUSD, SPXUSD, NSXUSD, UDXUSD, WTIUSD, BCOUSD, GRXEUR, UKXGBP, JPXJPY, FRXEUR, AUXAUD, HKXHKD.
- **Qualidade (só de dado, antes de qualquer estratégia):** um instrumento-mês entra se tiver ao menos 60% dos minutos da mediana do instrumento e nenhum preço não positivo. Um instrumento entra se sobrarem ao menos 36 meses. A lista final vai para `docs/forex/c3-data-manifest.csv`, commitada antes de a primeira estratégia rodar.
- **Barras:** D1 pelo dia FX (17:00 a 17:00 de Nova York, com o horário de verão de Nova York), H4 em blocos de 4 horas a partir das 17:00 de Nova York, H1 por hora cheia. Abertura do primeiro minuto, máxima, mínima e fechamento do último.

## Custos (fixos)

Os preços são só bid; o custo entra como um valor por operação completa (ida e volta), em unidades de preço, que soma spread, comissão e slippage da conta Zero da Fusion, com margem conservadora:

- Majors em pips: EURUSD e USDJPY 0,9; GBPUSD, AUDUSD, USDCAD, USDCHF 1,1; NZDUSD 1,3.
- Cruzados em pips: EURGBP 1,3; EURJPY 1,4; AUDJPY 1,4; EURCHF 1,5; CADJPY 1,6; AUDCAD 1,7; AUDCHF 1,8; AUDNZD 1,8; EURAUD 1,8; EURCAD 1,8; GBPJPY 1,8; NZDJPY 1,8; CHFJPY 2,0; CADCHF 2,0; NZDCAD 2,1; GBPCHF 2,2; GBPAUD 2,3; NZDCHF 2,3; GBPCAD 2,4; EURNZD 2,6; GBPNZD 3,3.
- Outros, em unidade de preço: XAUUSD 0,35; XAGUSD 0,035; SPXUSD 0,8; NSXUSD 2,5; GRXEUR 2,0; UKXGBP 1,5; FRXEUR 1,5; AUXAUD 1,5; JPXJPY 15; HKXHKD 12; WTIUSD 0,05; BCOUSD 0,05; UDXUSD 0,06.
- **Estresse:** o dobro do custo acima. Swap e financiamento: zero no cenário base (as famílias de D1 carregam posição por dias; é uma limitação conhecida, informada no relatório, e o critério exige resultado positivo no estresse também).

## Execução simulada

- Decisão no fechamento da barra do sinal; execução na abertura do primeiro minuto seguinte (preço bid; o custo cobre spread e slippage).
- Stop e alvo acompanhados minuto a minuto. Um minuto que abre além do stop executa na abertura (gap). Se stop e alvo cabem no mesmo minuto, vale o stop.
- Stop móvel e saídas por regra só mudam no fechamento de uma barra do timeframe da estratégia.
- Uma posição por instrumento e configuração; sinal oposto com posição aberta fecha a posição e abre a nova na mesma abertura.
- Posição aberta no fim de um trecho contínuo de dados fecha na última barra.

## Unidade de resultado

Como algumas estratégias originais não têm stop, o resultado de cada operação é medido em **ATR**: (movimento a favor − custo) ÷ ATR(20) do timeframe da estratégia no fechamento da barra do sinal. É o equivalente a arriscar sempre o mesmo múltiplo da volatilidade.

## Estratégias (fixas: as únicas; K = 14)

Parâmetros de quem publicou. Todas operam comprado e vendido.

- **TURTLE20 (D1):** Turtle Sistema 1 (Richard Dennis e William Eckhardt, como descrito por Curtis Faith). Entra quando o fechamento passa da máxima dos 20 dias anteriores (vendido: abaixo da mínima). Stop inicial 2 × ATR(20). Sai quando o fechamento passa da mínima dos 10 dias anteriores (vendido: máxima). Sem o filtro de "pular depois de um ganho" e sem aumentar posição.
- **TURTLE55 (D1):** Turtle Sistema 2: rompimento de 55 dias, saída pelo canal de 20 dias, stop 2 × ATR(20).
- **TSMOM (D1):** momentum de série temporal (Moskowitz, Ooi e Pedersen, 2012): posição igual ao sinal do retorno dos últimos 252 dias, revista a cada fechamento diário; troca de lado quando o sinal troca. Sem stop.
- **RSI2_D1, RSI2_H4, RSI2_H1:** RSI(2) de Larry Connors: comprado se o fechamento está acima da média de 200 barras e o RSI(2) fecha abaixo de 10; sai quando o fechamento passa acima da média de 5. Vendido no espelho (abaixo da média de 200, RSI(2) acima de 90, sai abaixo da média de 5). Sem stop, como no original. A versão D1 é a publicada; H4 e H1 são a mesma regra em barras menores.
- **SQUEEZE_D1, SQUEEZE_H4:** Bollinger Squeeze (John Bollinger): quando a largura das bandas BB(20, 2) está no menor valor das 125 barras, um fechamento fora da banda abre na direção dele. Stop na banda oposta da barra do sinal. Sai quando o fechamento cruza a média de 20.
- **NR7_D1, NR7_H4:** NR7 de Toby Crabel: se a barra tem o menor range das últimas 7, o próximo fechamento acima da máxima dela compra (abaixo da mínima vende), com stop no extremo oposto dela. Sai no fechamento da 3ª barra depois da entrada. O sinal vale só na barra seguinte à NR7.
- **MACD200_H1, MACD200_H4:** a estratégia de varejo mais repetida em vídeos: MACD(12, 26, 9) cruza a linha de sinal abaixo de zero com o fechamento acima da EMA 200 (compra; espelho para venda). Stop na mínima das 10 barras anteriores (vendido: máxima). Alvo 1,5 × a distância do stop.
- **CHANDELIER_D1, CHANDELIER_H4:** cruzamento de médias EMA 50 / EMA 200 com a saída Chandelier de Chuck LeBeau: stop móvel na máxima desde a entrada menos 3 × ATR(22) (vendido: mínima mais 3 × ATR(22)), só sobe (vendido: só desce). Troca de lado no cruzamento oposto.

Indicadores: médias simples e exponenciais pelo fechamento; RSI e ATR com a suavização de Wilder; canais e mínimas/máximas anteriores excluem a barra atual; tudo calculado só com barras já fechadas.

## Carteira

Informativa, não entra no critério: todas as operações das configurações aprovadas, somadas em ATR, com peso igual por operação, por instrumento e por dia.

## Critério

Estatística de cada configuração: média do resultado em ATR por operação e t com erro padrão por cluster de dia FX de entrada, como nas rodadas anteriores (`fx_fast_stats`).

- **Descoberta (grupo A, 2012-01 a 2026-08):** aprovada se todas: (1) valor-p ajustado da chave A (bootstrap estacionário de dias FX, bloco médio de 10 dias, 5.000 reamostragens, semente 20260924, máximo-t entre as 14) ≤ 0,05; (2) valor-p ajustado da chave B (placebo: o retorno de cada minuto troca de sinal com probabilidade 1/2, uma moeda por minuto igual para todos os instrumentos, 200 conjuntos, semente 20260925, máximo-t entre as 14) ≤ 0,05; (3) média > 0 no custo base; (4) média > 0 no estresse; (5) ao menos 60% dos instrumentos com média > 0; (6) ao menos 200 operações.
- **Calibração antes da descoberta:** os 200 conjuntos placebo do grupo A medem a falsa aprovação do critério inteiro (cada conjunto contra os outros 199), limite 8%. Relatório commitado antes da descoberta.
- **Réplica (grupo B, 2012-01 a 2026-08), só as aprovadas, sem mudar nada:** média > 0 no custo base e no estresse; valor-p unilateral do t por cluster ≤ 0,05 com correção de Holm entre as aprovadas; ao menos 55% dos instrumentos do grupo B com média > 0; ao menos 200 operações.
- **Confirmação (prospectiva, uma vez):** as replicadas, em dados de todos os 41 instrumentos com entrada depois do commit deste documento, quando existirem ao menos 100 operações; mesmas regras da réplica.
- Se nada passar numa etapa: paro e levo os números ao Igor. Nenhuma estratégia desta rodada vai para demo antes da réplica.

## O que NÃO será feito

- Trocar parâmetro, timeframe, mercado, custo ou critério depois de ver resultado.
- Rodar qualquer estratégia no grupo B antes do relatório de descoberta commitado.
- Tirar do grupo B um mercado que perdeu, ou somar grupos para salvar uma etapa.

## O que se espera (palpite registrado antes do dado)

Seguir tendência (TURTLE, TSMOM, CHANDELIER) em D1 é a família com mais evidência publicada, mas em FX a vantagem foi fraca na década de 2010; RSI2 e MACD200 em barras curtas devem perder para o custo. Palpite: 15% a 25% de chance de alguma configuração passar descoberta e réplica, quase toda ela nas famílias de tendência em D1.
