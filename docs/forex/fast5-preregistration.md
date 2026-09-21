# Pré-registro, rodada 5: entrada probabilística com breakeven e runner

Registrado em 2026-09-21 antes de calcular qualquer label, probabilidade ou resultado desta rodada.
O objetivo é testar exatamente a arquitetura pedida pelo Igor: um modelo decide **se e para qual
lado entrar**; um gestor determinístico protege a operação e deixa o vencedor correr. Esta rodada é
pesquisa offline. Não envia ordens, não altera Demo ou LIVE e não acessa a semana da corretora.

## Hipótese

Para cada par, as features causais de cotações da rodada 4 conseguem estimar a probabilidade de o
preço executável andar pelo menos **+0,5R líquido** antes de tocar **-1R**. Selecionando apenas as
maiores probabilidades e, depois da ativação, substituindo o risco por breakeven líquido mais
trailing sem alvo, os poucos runners podem pagar os stops e timeouts das demais entradas.

O modelo não tenta prever o preço final. Ele responde: “qual a chance de esta entrada ganhar espaço
suficiente para ser protegida?”. A gestão da posição não usa IA: é uma regra auditável, idêntica no
backtest e no futuro runner.

## Amostras e blindagem

Fonte: os mesmos ticks bid/ask HistData e as mesmas 51 colunas causais da rodada 4. Volume continua
fora porque é zero. Duplicatas, crossed quotes, gaps, eventos e custos seguem a fábrica validada da
rodada 4.

- treino do classificador: 2019-01 a 2020-12;
- early stopping e calibração de probabilidade: 2021-Q1;
- escolha de threshold e distância do trailing: 2021-Q2;
- validação de desenvolvimento: 2021-H2, aberta somente depois de a política ser congelada;
- confirmação cega: 2022, aberta uma vez somente se 2021-H2 passar;
- D4 (2023-01 a 2024-08) e S3 (2024-09 a 2026-05) continuam protegidos e exigem novos gates;
- a semana da Fusion/cTrader continua reservada para depois de D4 e S3 aprovados.

Purge e embargo de seis horas são aplicados nas fronteiras. Resultados já vistos das rodadas
anteriores não são confirmação desta hipótese; por isso 2022 é o primeiro blind test possível.

## Entrada e unidade de risco

O evento e a entrada são os mesmos da rodada 4: decisão depois de 512 cotações e a cada 128 updates,
fill na primeira cotação posterior. Compra entra no ask; venda entra no bid. O risco `1R` é o maior
entre o range causal das 256 cotações anteriores e quatro vezes o custo completo base.

Para cada par existem dois classificadores direcionais, long e short. Cada um estima separadamente:

- `P_base`: probabilidade de alcançar +0,5R líquido antes de -1R em até 600 segundos;
- `P_stress`: a mesma probabilidade com spread dobrado e slippage de estresse.

O score é `min(P_base, P_stress)`. Se os dois lados passarem o corte no mesmo evento, entra somente o
lado com maior score. Enquanto houver posição aberta naquele par, todo novo evento é ignorado.

## Modelo e tentativas

Um XGBoost binário por par, lado e cenário, com configuração fixa: profundidade 3, learning rate
0,03, até 700 árvores, subsample/colsample 0,8, `min_child_weight=100`, `reg_lambda=10` e early
stopping de 50 árvores. A probabilidade é calibrada por Platt apenas em 2021-Q1.

Thresholds pré-declarados: `{0,55; 0,60; 0,65; 0,70}`. Todos contam como tentativas, inclusive os que
não operarem. Não serão adicionados indicadores ou hiperparâmetros depois de ver 2021-Q2/H2.

## Gestor da posição

Antes da ativação:

- stop inicial em -1R líquido;
- se +0,5R não for atingido em 600 segundos, saída a mercado no primeiro tick do prazo;
- se -1R vier antes, stop com slippage.

Depois da ativação:

- o piso sobe imediatamente para **0R líquido**, já cobrindo spread, comissão e slippage;
- não existe take-profit;
- o stop acompanha o melhor lucro executável observado, nunca recua;
- distâncias candidatas pré-declaradas: `{0,5R; 1,0R; 1,5R}`;
- `stop = max(0R, melhor_R - distância)` para long e short;
- timeout total de seis horas; gap superior a 120 segundos invalida o exemplo;
- custos base e estresse são simulados separadamente nos preços executáveis.

São 12 combinações threshold × trailing por par. A melhor de 2021-Q2 é congelada para cada par antes
de 2021-H2. Um par pode legitimamente escolher não operar.

## Gates

Uma política por par só pode seguir de 2021-Q2 para 2021-H2 com pelo menos 30 trades, média base e
estresse positivas, ao menos dois terços dos meses positivos e profit factor de estresse acima de
1,05. O ranking usa primeiro média de estresse, depois limite inferior do bootstrap e, por fim,
menos trades como desempate conservador.

O portfólio congelado só abre 2022 se em 2021-H2 tiver:

- média base e estresse positivas;
- limite inferior unilateral de 95% do bootstrap por dia FX acima de zero no base;
- pelo menos 300 trades, seis pares positivos e 55% dos meses positivos;
- nenhum par responsável por mais de 35% do lucro;
- PBO <= 0,20 e DSR >= 0,95 contando todas as tentativas.

Em 2022, os mesmos gates se repetem com mínimo de 500 trades. Falha encerra a rodada sem abrir D4.

## Testes obrigatórios

- ordem exata dos toques: +0,5R antes do stop não pode ser inferido apenas por MFE/MAE;
- simetria long/short e preço executável bid/ask;
- breakeven verdadeiramente líquido depois de todos os custos;
- stop nunca recua e trailing captura runners maiores que qualquer alvo fixo;
- timeout pré e pós-ativação, gap e próximo tick;
- causalidade, fronteiras temporais, purge/embargo e recusa de 2022/D4/S3;
- calibração de probabilidade e relatório por par, lado, mês e motivo de saída;
- hashes de fábrica, labels, modelos, políticas e relatórios em registro append-only.

## Ordem

1. Commit deste pré-registro e do guard.
2. Implementar e testar labels de primeiro toque e simulador de trailing em dados sintéticos.
3. Materializar somente 2019-2021 e registrar hashes.
4. Treinar por par/lado, escolher políticas apenas em 2021-Q2 e congelá-las.
5. Rodar 2021-H2 uma vez.
6. Abrir 2022 somente se os gates passarem.
7. D4, S3, semana da corretora e qualquer canário permanecem fechados até suas portas específicas.

## Emenda operacional 1 — agenda comum entre cenários

Registrada antes de calcular qualquer resultado de trailing em 2021-Q2. O pré-registro exige uma
posição por par, mas não explicitava como manter a mesma amostra quando custo base e estresse geram
horários de saída diferentes. Para impedir que o cenário escolha trades diferentes, cada candidato
só é elegível se tiver trajetória válida nos dois cenários; depois de entrar, o próximo sinal só
pode ser considerado após o **maior** dos dois tempos de saída. Assim base e estresse avaliam as
mesmas entradas e nenhuma delas contém sobreposição. É uma decisão conservadora e não altera
thresholds, distâncias, gates ou quantidade de tentativas.
