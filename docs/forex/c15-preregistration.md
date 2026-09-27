# Pré-registro C15: o prêmio do dia de FOMC nas moedas e a compra do S&P 500 no pico do medo

Registrado em 2026-09-27, antes do código e de qualquer resultado desta rodada. As hipóteses saíram de quatro pesquisas feitas no mesmo dia: literatura acadêmica, comunidades de traders, dados alternativos gratuitos e informações novas para o modelo. Duas candidatas fortes da literatura já estavam testadas e foram deixadas de fora:

- o rebalanceamento de hedge no fix de fim de mês condicionado às bolsas (Melvin e Prins), reprovado no C11;
- a deriva noturna dos índices, reprovada no C11 e no C12.

## H1 — prêmio do dia de FOMC (Mueller, Tahbaz-Salehi e Vedolin, Journal of Finance, 2017)

**Mecanismo.** Os intermediários têm capital limitado e cobram um prêmio para carregar risco de política monetária. Esse prêmio aparece no dia do anúncio do FOMC, nas moedas de juro alto contra o dólar.

**Evidência.**

- No artigo, com dados de 1994 a 2013, a carteira de moedas de juro alto rende 14,5 pb no dia de FOMC, contra 1,7 pb nos outros dias. Líquido do bid-ask, são 12,2 pb por evento (t 2,1).
- Uma replicação independente até 2025 indica que o efeito continua.
- Lee e Wang (2025) mostram que parte do efeito reverte entre 12 e 24 horas depois. Por isso a saída é no mesmo dia.

**Regra (fixa)**

- **Dias:** as 188 decisões agendadas de 2003 a 2026, listadas em `docs/forex/c15-fomc-dates.csv`.
  - De 2003 a 2018, as datas vêm das páginas históricas do federalreserve.gov.
  - De 2019 em diante, vêm do calendário oficial já verificado no repositório.
  - As reuniões extraordinárias (2020-03-03 e 2020-03-15) ficam de fora.
- **Moedas**
  - Das sete (EUR, GBP, JPY, CHF, AUD, NZD e CAD), entram as que têm juro de 3 meses acima do americano no dia. A fonte é o FRED IR3TIB01, lido no mês M−2, como no C13 e no C14.
  - Todas com o mesmo peso, compradas contra o dólar: EURUSD, GBPUSD, AUDUSD e NZDUSD comprados; USDJPY, USDCHF e USDCAD vendidos.
  - Se nenhuma tiver juro acima do americano, não opera.
- **Janela**
  - Entrada na primeira barra M1 a partir das 16:00 de Nova York do dia útil anterior, com no máximo 5 minutos de atraso.
  - Saída no fechamento da barra das 15:59 do dia da decisão, que tem de existir até 5 minutos antes.
  - Sem qualquer uma das duas barras, a perna não conta.
- **Stop de proteção**
  - 1 ATR diário: o ATR(14) dos dias FX completos anteriores.
  - É checado minuto a minuto e, dentro do minuto, antes de qualquer outra coisa. Um gap executa na abertura.
  - Existe porque nada roda na demo sem stop no servidor.
- **Custo:** o do C3 por par, ida e volta. No estresse, o dobro.
- **Swap**
  - É cobrado um rollover. Na quarta-feira, ele conta 3.
  - O cálculo segue o método do C14: juros do FRED com markup de 1% ao ano.
  - No estresse, o markup é de 2% e o crédito positivo é zerado.
- **Unidade:** pontos-base do nocional, pela média das pernas de cada evento. O resultado em R (1 ATR) é só informativo.

**Julgamento**

- **Amostra principal:** 2014-01 a 2026-08, depois do fim da amostra do artigo. São cerca de 100 eventos, com os dados M1 do C3 (manifesto do C3).
- **Aprovada se todas as condições valerem**
  - média líquida > 0, com custo normal e no estresse;
  - p unilateral ≤ 0,05, pelo t calculado sobre os eventos;
  - média maior que a da mesma janela nos outros dias úteis sem FOMC que caem no mesmo dia da semana;
  - **placebo:** em 20 cópias em que as datas de FOMC são trocadas por dias úteis sorteados sem FOMC (a mesma quantidade por ano, os mesmos dias da semana), no máximo 1 cópia alcança a média real.
- **Informativo (não decide):** 2003–2013, que se sobrepõe ao artigo. Os dados são o M1 do manifesto do C10 até 2011 e o do C3 em 2012–2013.
- **Poder, escrito antes**
  - O efeito esperado é de 8 a 15 pb por evento, com desvio perto de 50 pb.
  - Com cerca de 100 eventos, o t esperado fica entre 1,6 e 3,0.
  - Uma reprovação pode ser falta de poder, e isso será dito.

## H2 — compra do S&P 500 no pico do VIX (sinal contrário)

**Referências:** Simon e Wiggins (2001); Bollerslev, Tauchen e Zhou (2009).

**Mecanismo.** No pico do medo, o seguro fica caro: o prêmio de risco sobe, e o mercado tende a devolvê-lo nas semanas seguintes.

**Regra (fixa)**

- **Sinal:** no fechamento do pregão t, VIX(t) ≥ 1,25 × a média do VIX nos 20 pregões anteriores (de t−20 a t−1).
- **Operação**
  - Compra do S&P 500 (^GSPC) no fechamento de t e venda no fechamento de t+10 pregões.
  - Um sinal que aparece com a posição aberta é ignorado.
- **Custo**
  - 1 pb ida e volta (cerca do spread do US500 na Fusion); no estresse, 2 pb.
  - Financiamento do índice comprado: 6% ao ano sobre os dias corridos carregados; no estresse, 9%.
- **Dados:** VIX da CBOE (`VIX_History.csv`, diário, desde 1990) e o ^GSPC diário do Yahoo (`data/c5`).

**Julgamento**

- **Amostra:** 2003-01 a 2026-08 inteira, nunca testada com esta regra. Uma rodada.
- **Aprovada se todas as condições valerem**
  - média líquida por operação > 0, com custo normal e no estresse;
  - p unilateral ≤ 0,05, pelo t sobre as operações;
  - média maior que a de todas as janelas de 10 pregões do período (o S&P comprado sempre, com o mesmo custo);
  - pelo menos 60% dos anos que tiveram operação terminando positivos.

## Correção para testes múltiplos

Holm entre H1 e H2.

## Depois

- **Aprovada:** vai para a demo 10139135 com risco pequeno (de 0,5% a 1% do patrimônio por ATR), como o Gotobi. A demo é o juiz final.
- **Sem aprovada:** relatório e nada na demo.
- **Emendas:** só antes de qualquer resultado, registradas em `docs/forex/c15-registry.jsonl`.
