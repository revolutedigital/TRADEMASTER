# Pré-registro C13: banca de day trade (A) e prêmios de juro e tendência em moedas (C)

Registrado em 2026-09-26, antes de qualquer código ou resultado desta rodada. O período 2003-01 a 2011-12 em M1 (`backend/data/c10/holdout_m1`, manifesto `docs/forex/c10-holdout-manifest.csv`) segue intocado por resultados, e esta rodada o usa **uma única vez**, para as duas frentes juntas.

## Por que

C10 a C12 mostraram duas coisas:

- As regras de day trade não têm vantagem depois do custo, com nenhum formato de saída.
- Parte dos padrões existe antes do custo.

Sobram duas perguntas com base real:

- **(A)** uma banca que aprende *quando* entrar, pelo horário, pela volatilidade, pelo custo relativo ao movimento, pela tendência e pelo desempenho recente de cada regra, consegue separar as operações boas das ruins no day trade? É a ideia do Igor: a vantagem de quem opera há 20 anos está em saber quando não operar.
- **(C)** os prêmios mais documentados em moedas (juro, tendência), segurados por dias ou semanas com o swap histórico, são positivos? **C não é day trade.** Entra porque é onde a literatura mostra vantagem, e fica identificada assim no relatório.

## Frente A: banca de day trade

**Operações candidatas:** os 34 conjuntos família-parâmetro de C10 (`fx_c10_families`), nos 37 mercados. Todos usam a mesma gestão fixa: stop de 1 ATR diário, alvo de 3 R, até o fim do dia FX. É a linha da grade de C10 (depois da emenda 1) mais frequente entre as melhores de cada conjunto. Escolher a gestão olhando C10 é seleção, e quem julga é o período intocado. Execução, custo e unidade são os de C10: uma posição por vez por conjunto e mercado, R líquido, e estresse com custo dobrado.

**Avaliadores (todos conhecidos no fechamento da barra do sinal).** "Lado" multiplica os direcionais:

- conjunto, lado, classe do mercado, minutos da barra;
- hora no relógio da bolsa, dia da semana, minutos desde a abertura da sessão, dentro ou fora da sessão;
- custo ÷ ATR diário;
- ATR(14) ÷ ATR(100) diários;
- faixa do dia FX até ali ÷ ATR;
- faixa de ontem ÷ ATR;
- distâncias com lado, em ATR: à abertura do dia, ao fechamento de ontem, à máxima e à mínima de ontem;
- tendência com lado: (fechamento de ontem − média de 50 dias) ÷ ATR;
- retornos com lado de 1, 3 e 12 barras ÷ ATR;
- RSI(14) − 50 com lado;
- z de Bollinger(20) com lado;
- faixa da barra ÷ média das 20 anteriores;
- média de R das últimas 20 operações **já fechadas** do mesmo conjunto no mesmo mercado;
- média de R das últimas 200 operações já fechadas do mesmo conjunto em todos os mercados.

**Modelo:** `XGBRegressor` prevendo o R líquido, com 300 árvores, taxa 0,05, subsample e colsample 0,8, `tree_method=hist` e semente fixa. Opera se a previsão for maior que τ. Grade de 8 combinações: profundidade {3, 5} × peso mínimo por folha {200, 2000} × τ {0; 0,05}.

**Walk-forward dentro de 2012-01 a 2026-08:** para cada ano de validação de 2016 a 2026 (2026 até agosto), o modelo treina com as operações que **saíram** antes do início do ano e decide as operações que entram nele. A combinação escolhida é a de maior média de R das operações aceitas, somando os 11 anos, com pelo menos 3.000 aceitas.

**Passa para o período intocado se, na combinação escolhida:**

- média > 0 com custo normal e com custo dobrado;
- p unilateral pelo t agrupado por dia ≤ 0,05 ÷ 8 (correção pelas 8 combinações);
- ≥ 55% dos mercados com 20+ aceitas positivos.

**Modelo final:** treinado com todas as operações que saíram até 2026-08-31 e congelado, com o hash no relatório. No período intocado, as operações candidatas e os avaliadores são gerados do mesmo jeito, só com o próprio período.

## Frente C: juro e tendência em moedas (segura dias ou semanas)

**Universo:** as 7 moedas contra o dólar pelos pares diretos. EUR, GBP, AUD e NZD entram por EURUSD, GBPUSD, AUDUSD e NZDUSD. JPY, CHF e CAD entram por USDJPY, USDCHF e USDCAD, com o sinal invertido. O preço é o fechamento do dia FX (17:00 de Nova York), montado do M1.

**Juro:** taxa interbancária de 3 meses da OCDE no FRED, `IR3TIB01{US,EZ,GB,JP,CH,AU,NZ,CA}M156N`, mensal. O dia do mês M usa o valor do mês M−2, para ser conhecido de verdade na data. Depois da última observação, o valor segue o último.

**Retorno em excesso** de estar comprado na moeda k contra o dólar, de um fechamento ao seguinte: log-retorno do preço (no sentido "k sobe") + (r_k − r_USD − markup) × dias corridos ÷ 365. Vendido: o contrário do preço + (r_USD − r_k − markup) × dias ÷ 365. **Markup** de 1% ao ano por lado, o que cobre o que a Fusion cobra hoje, medido em 2026-09-25 (0,2% a 1,4% ao ano sobre o diferencial). No estresse, 2%.

**Custo de execução:** o custo de C3 do par, metade na abertura e metade no fechamento de cada posição; uma virada paga as duas metades. No estresse, o dobro.

**As 4 estratégias.** A exposição bruta é 1: a soma de |peso| = 1.

- **CARRY** (Lustig, Roussanov e Verdelhan 2011): no fechamento do primeiro dia FX de cada semana, ordena as 7 moedas pelo juro e fica comprada nas 2 de maior juro e vendida nas 2 de menor (0,25 cada).
- **CARRY_TREND:** igual, mas cada perna só existe se o retorno do preço em 63 dias (a favor dela) for positivo; perna vetada fica com peso 0.
- **TSMOM** (Moskowitz, Ooi e Pedersen 2012): no primeiro dia FX de cada mês, cada moeda recebe o sinal do seu retorno em excesso de 252 dias, com peso proporcional a 1 ÷ volatilidade diária de 60 dias, normalizado para soma de |peso| = 1.
- **DOLLAR_CARRY** (Lustig, Roussanov e Verdelhan 2014): no primeiro dia FX de cada mês, compra as 7 moedas contra o dólar (1/7 cada) se o juro médio delas for maior que o do dólar; senão, vende as 7.

O peso decidido num fechamento vale a partir do dia seguinte. **Estatística:** retorno diário da carteira; t e bootstrap estacionário de máximo t sobre as 4 (5.000 sorteios, bloco médio de 10 dias, semente 20261021).

**Passa para o período intocado se:** p ajustado ≤ 0,05; média > 0 com custo normal e no estresse; positiva em ≥ 60% dos anos.

## Período intocado: 2003-01 a 2011-12, uma rodada para as duas frentes

- As candidatas das duas frentes rodam uma vez. Pela frente C, os dias de 2003 a 2011 dos 7 pares; pela A, as operações dos 37 mercados que o manifesto inclui.
- p unilateral: para A, pelo t agrupado por dia; para C, pelo t dos retornos diários.
- **Holm** sobre todas as candidatas levadas, das duas frentes juntas.
- **Aprovada:** p de Holm ≤ 0,05; média > 0 com custo normal e no estresse. Para A, também ≥ 300 operações aceitas e ≥ 55% dos mercados (com 20+) positivos. Para C, também positiva em ≥ 5 dos 9 anos.

## Depois

Aprovada vai para a demo 10139135: a banca de day trade como um robô novo; C como carteira com rebalanceamento. O sizing é o de sempre: 0,5% do capital por ATR diário, ou o volume mínimo. Sem aprovada, o relatório diz isso com os números: taxa de acerto, operações por dia, retorno anual, pior queda.

**Proibido:** olhar métrica de resultado antes deste texto e do código estarem commitados; mudar gestão, avaliadores, grade, estratégias, custos ou critérios depois de ver resultado; rodar o período intocado mais de uma vez. Emendas só antes de resultado, em `c13-registry.jsonl`.
