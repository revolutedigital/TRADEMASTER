# Pré-registro, rodada 6: fila global calibrada de entradas com runner

Registrado em 2026-09-21 depois de observar somente as probabilidades e contagens da rodada 5 nos
sete primeiros pares, mas **antes de calcular qualquer resultado financeiro de trailing em
2021-Q2**. A rodada 5 mostrou que o requisito absoluto de 55% não autoriza trades: nos sete pares
concluídos o maior score conservador ficou abaixo desse corte. Isso não testa a hipótese econômica
central, pois uma entrada com probabilidade menor ainda pode ter valor se a cauda dos runners for
grande.

Emenda solicitada pelo Igor antes de qualquer resultado financeiro de trailing: o ranking deixa de
ser por par. `Top P` passa a significar as melhores oportunidades **globais e causais**, comparadas
entre todos os mercados disponíveis. A extração anterior foi interrompida em 24 de 30 partições e
nenhuma tentativa de trailing foi executada. Partições completas permanecem reutilizáveis; a
partição interrompida não entrou no manifesto.

## Elementos herdados e congelados

- mesmos ticks bid/ask, limpeza, 50 features causais e eventos da rodada 5;
- mesmos labels, custos e unidade de risco da rodada 5;
- treino 2019-2020, calibração 2021-Q1, seleção 2021-Q2 e validação 2021-H2, com purge/embargo de
  seis horas;
- stop inicial -1R líquido, ativação +0,5R em até 600 segundos, breakeven líquido imediato,
  trailing sem alvo a 0,5R, 1R ou 1,5R e timeout total de seis horas;
- a agenda comum usa o maior horário de saída entre base e estresse, mantendo as mesmas entradas
  sem sobreposição nos dois cenários;
- 2022, D4, S3 e semana da corretora permanecem fechados.

## Modelos local e global

O classificador local continua separado por par, lado e cenário. Em paralelo, dois classificadores
globais, base e estresse, aprendem todos os pares e os dois lados com identificação one-hot do par e
features direcionais orientadas para o lado da entrada. Para manter memória e dependência temporal
sob controle, o global usa stride determinístico 4 com offsets diferentes por par/lado. A
configuração dos XGBoost fica congelada em depth 3, learning rate 0,03, até 700 árvores,
subsample/colsample 0,8, `min_child_weight=100`, `reg_lambda=10` e early stopping 50.

O Q1 é dividido temporalmente, sempre com seis horas de embargo:

- 2021-01-01 a 2021-02-15: early stopping;
- 2021-02-15 a 2021-03-15: ajuste da calibração;
- 2021-03-15 a 2021-04-01: auditoria da calibração e escolha do blend.

Platt, isotônica e beta calibration são ajustadas no bloco de calibração. O método e o peso do
blend local/global em `{0; 0,25; 0,5; 0,75; 1}` são escolhidos exclusivamente pela menor Brier
score na auditoria, com log loss como desempate. A probabilidade conservadora é
`min(P_base, P_stress)`. A auditoria também produz bins de confiabilidade e um limite inferior de
Wilson de 95%; o `trusted_score` é o menor entre a probabilidade pontual e esse limite. O modelo não
é elegível se não melhorar o Brier da taxa-base na auditoria.

## O que significa Top P

Primeiro permanece somente o lado de maior `trusted_score` por par e timestamp. Depois todos os
pares entram numa única fila. Para cada dia FX, o corte Top P é calculado usando somente os scores
dos 60 dias corridos anteriores; o próprio dia e qualquer futuro ficam fora. Também existe um piso
absoluto, calculado no fim de Q1 para a mesma fração P. O corte efetivo é o maior entre o quantil
móvel passado e o piso de Q1. Portanto, estar no topo de um dia ruim não força operação.

Frações candidatas: `{0,25%; 0,5%; 1%; 2%; 5%; 10%}`. Elas são combinadas às três distâncias de
trailing, totalizando **18 tentativas globais**, não 18 por par. No máximo três posições podem ficar
abertas simultaneamente no portfólio e somente uma por par. Quando base e estresse saem em momentos
diferentes, a vaga permanece ocupada até o horário mais tardio.

O protocolo é extensível a outros mercados porque preço, custos e resultado são normalizados em R,
mas esta rodada comprova somente os dez pares com ticks já disponíveis. Um novo ativo precisa de
histórico bid/ask, modelo de custos e auditoria de calibração próprios antes de competir na fila.

## Escolha e validação

O gate Q2 exige pelo menos 100 trades, médias base e estresse positivas, dois dos três meses
positivos, profit factor de estresse acima de 1,05, quatro pares participantes e nenhum par com mais
de 35% do lucro positivo. O ranking das políticas que passarem usa média de estresse, limite
inferior de bootstrap e menos trades.

Depois de congelar uma política por par, 2021-H2 é aberto uma única vez. O portfólio só pode abrir
2022 com média base e estresse positivas, bootstrap unilateral de 95% acima de zero no base, pelo
menos 300 trades, seis pares positivos, 55% dos meses positivos, concentração máxima de 35%,
PBO <= 0,20 e DSR >= 0,95 considerando as 18 tentativas. Falha encerra a rodada em 2021-H2.

Esta rodada é pesquisa offline: não ativa estratégia e não envia ordens Demo ou LIVE.
