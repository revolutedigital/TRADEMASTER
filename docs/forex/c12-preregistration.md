# Pré-registro C12: a deriva noturna do índice americano, confirmada fora da amostra

Registrado em 2026-09-25, antes de baixar ou olhar os dados desta rodada.

## De onde vem a hipótese (e por que ela precisa de confirmação)

Em C11 (2012–2026), comprar S&P 500 e Nasdaq 100 das 18:05 às 09:25 de Nova York rendeu +0,027 R por noite antes do custo (t ≈ 4,5), e −0,001 R depois do custo de C3 (0,8 ponto no S&P, 2,5 no Nasdaq). O spread real da Fusion lido na demo em 2026-09-25 foi 0,36 ponto no US500 e 0,8 no NAS100, sem comissão em índice. Com ele a janela seria positiva. **Essa conclusão saiu de olhar C11, então não vale como prova.** A prova tem que vir de dado que nenhuma rodada usou para isso. A literatura é a mesma: Cliff, Cooper e Gulen (2008), e Boyarchenko, Larsen e Whelan (2023).

Posição aberta das 18:05 às 09:25 não atravessa as 17:00 de Nova York, então não paga swap. Um índice comprado 24 horas na Fusion paga de 3% a 9% ao ano.

## Teste de confirmação (uma rodada)

- **Dados:** SPY e QQQ diários do Yahoo, 2000-01-03 a 2011-12-30. São ETFs, e a abertura deles é negócio de verdade (a abertura do índice à vista é defasada). O retorno noturno é a abertura do dia sobre o fechamento da véspera. Esses ETFs e essa pergunta não entraram em nenhuma rodada. C5 usou os índices diários de 2000–2011 para outras regras.
- **Qualidade:** noite com retorno acima de 20% em módulo é erro de dado (desdobramento) e sai. As cotações não são ajustadas por dividendo, então a data-ex puxa o retorno noturno para baixo em cerca de 2% ao ano. Isso joga contra a hipótese e fica como está.
- **Custo por noite, ida e volta:** 1 ponto-base (≈ o custo de C3 no S&P) e, no estresse, 2 pontos-base.
- **Estatística:** média do retorno noturno líquido; t agrupado por dia (as duas ETFs no mesmo dia são um grupo); p unilateral.
- **Confirmada se:** p ≤ 0,05 com 2 pontos-base; média > 0 com 1 e com 2 pontos-base; as duas ETFs positivas; média positiva em pelo menos 7 dos 12 anos.

## Se confirmar

A janela vai para a demo 10139135: compra US500 e NAS100 às 18:05 e vende às 09:25 de Nova York, dias úteis, 0,5% do capital por ATR diário ou o volume mínimo, stop de proteção de 1 ATR. A demo mede o custo real de cada execução (spread e deslize) contra os 0,36 e 0,8 pontos usados aqui. A conclusão sobre o dinheiro sai da demo.

## Se não confirmar

A janela não vai para a demo, e o relatório diz isso.
