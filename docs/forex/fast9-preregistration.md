# Pré-registro, rodada 9: expectativa líquida do gestor completo

Registrado em 2026-09-21 antes de materializar qualquer payoff de treino desta rodada. Esta é uma
pesquisa offline. H2, 2022, D4, S3, semana da corretora, Demo e LIVE continuam fechados.

## Hipótese

As rodadas 6–8 calibraram a chance de primeiro toque, mas esse evento não representou o resultado
econômico completo. A nova hipótese é que as mesmas features causais podem ordenar diretamente a
distribuição do retorno líquido produzido por uma gestão fixa e, principalmente, reconhecer a ação
`não operar`.

Isto não repete as regressões terminais das rodadas 3–4. O alvo agora é o payoff exato e
path-dependent do gestor: stop inicial, ativação, breakeven, trailing e saída por tempo, calculados
tick a tick com os custos de cada cenário.

## Gestor congelado antes dos novos labels

- stop inicial em -1R;
- ativação em +0,1R dentro de 600 segundos;
- após ativar, piso em zero e trailing de 0,5R sobre o melhor resultado líquido;
- máximo de seis horas;
- custos `FUSION_ZERO` no base e o cenário `STRESS` já versionado;
- entrada no primeiro tick posterior, no ask para compra e bid para venda;
- no máximo três posições globais e uma por par na avaliação.

Esses parâmetros vêm da família já encerrada e contam como conhecimento gasto. Não serão
reotimizados nesta rodada.

## Dados, labels e divisões

Fonte: painel causal e ticks já materializados para os dez pares. Para conter custo computacional,
o treino de 2019–2020 usa deterministicamente um de cada quatro eventos, com offset fixo por par.
2021-Q1 usa todos os eventos válidos. Os dois lados e os cenários base/stress são simulados pelo
mesmo kernel exato.

- treino: 2019-01-01 até 2020-12-31, com stride 4;
- early stopping: 2021-01-01 a 2021-02-15;
- calibração: 2021-02-15 a 2021-03-15;
- auditoria: 2021-03-15 a 2021-04-01;
- gate de desenvolvimento: 2021-Q2;
- purge/embargo de seis horas em cada fronteira.

Q1 e Q2 não são apresentados como confirmação virgem: ambos já participaram das famílias
anteriores, e a escolha do gestor conhece os resultados agregados de Q2. A primeira verificação
prospectiva possível continua sendo 2021-H2, aberta somente se a política passar o gate Q2.

## Modelos declarados

Um painel global inclui as features causais, identidade do par e lado orientado. Para cada cenário
base/stress serão comparadas duas estimativas:

1. **direta:** XGBoost raso para E[retorno líquido em R];
2. **hurdle:** classificador para P(ativar), regressão de E[R | ativou] e regressão de
   E[R | não ativou], combinados pela lei da expectativa total.

Todos usam profundidade 3, learning rate 0,03, até 700 árvores, subsample/colsample 0,8,
`min_child_weight=100`, `reg_lambda=10` e early stopping. São permitidos blends
direta/hurdle `{0; 0,25; 0,5; 0,75; 1}`.

Cada blend é calibrado na janela de calibração por uma das transformações pré-declaradas:
identidade, affine OLS ou isotônica monotônica. A combinação é escolhida somente pelo menor MSE na
auditoria, com desempate por MAE e maior peso no hurdle. Predições calibradas são limitadas a
[-1,25R; +2R] para impedir extrapolação econômica absurda.

## Elegibilidade e `trusted_ev`

Um cenário só é elegível se, na auditoria:

- MSE for menor que prever sempre a média da própria auditoria;
- erro absoluto da média prevista for no máximo 0,05R;
- correlação de Spearman entre previsão e retorno for positiva;
- o decil superior tiver retorno realizado maior que a média total.

Em dez bins de quantil, calcula-se a média realizada e seu limite inferior unilateral de 95%. O
`trusted_ev` preserva a ordem dentro do bin e subtrai da previsão a diferença entre a média prevista
e esse limite. O score final é o menor `trusted_ev` entre base e stress. Se qualquer cenário falhar
a elegibilidade, Q2 não é simulado.

## Top P global e tentativas

Para cada dia FX de Q2, o corte usa somente os 60 dias anteriores. O piso absoluto vem de Q1 e é o
máximo entre zero e o quantil correspondente. São cinco frações globais:
`{0,25%; 0,5%; 1%; 2%; 5%}`. Apenas scores estritamente positivos podem operar. Se os dois lados
passarem no mesmo instante/par, vence o maior `trusted_ev`.

São cinco novas tentativas; o total acumulado das famílias 6–9 passa a 29. O gestor, custos e agenda
de posições são idênticos em todas elas.

## Gate Q2 e proteção de amostras

Uma política exige simultaneamente:

- pelo menos 100 trades;
- média base e stress positivas;
- limite inferior bootstrap de 95% da média base positivo;
- pelo menos dois dos três meses positivos;
- profit factor stress acima de 1,05;
- ao menos quatro pares participantes;
- nenhum par responsável por mais de 35% do lucro positivo.

Se mais de uma passar, vence a maior média stress, depois maior limite bootstrap e menos trades.
Nenhuma aprovação permite ordem real: ela abre somente uma execução congelada em 2021-H2. Falhar
em Q2 encerra esta família sem olhar H2 ou 2022.

## Emenda: funil progressivo de custo

Registrada em 2026-09-21 depois de concluir os labels de EURUSD e parte de GBPUSD, antes de treinar
qualquer modelo ou observar qualquer resultado de auditoria da rodada 9. A materialização integral
foi interrompida por custo excessivo de replay.

Antes do painel global, a hipótese passa por dois pilotos de aprendibilidade sem abrir Q2:

1. **P0, EURUSD:** usa somente as três partições EURUSD já concluídas e as mesmas janelas temporais.
2. **P1, três pares:** só será materializado se P0 passar; acrescenta GBPUSD e GBPJPY.

Para avançar, base e stress precisam, separadamente, ter MSE pelo menos 1% menor que o baseline da
média, Spearman acima de 0,02, erro absoluto de média de no máximo 0,05R e decil superior realizado
pelo menos 0,05R acima da média total. O modelo selecionado continua sendo direta versus hurdle e
seus blends/calibradores já declarados. Falhar em qualquer cenário encerra a hipótese imediatamente.

Somente P1 aprovado autoriza materializar os dez pares e executar o Top P no Q2. Os pilotos não
selecionam threshold de operação, não simulam portfólio e não abrem amostra protegida.
