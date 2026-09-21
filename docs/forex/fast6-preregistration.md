# Pré-registro, rodada 6: melhores entradas por ranking com runner

Registrado em 2026-09-21 depois de observar somente as probabilidades e contagens da rodada 5 nos
sete primeiros pares, mas **antes de calcular qualquer resultado financeiro de trailing em
2021-Q2**. A rodada 5 mostrou que o requisito absoluto de 55% não autoriza trades: nos sete pares
concluídos o maior score conservador ficou abaixo desse corte. Isso não testa a hipótese econômica
central, pois uma entrada com probabilidade menor ainda pode ter valor se a cauda dos runners for
grande. Esta nova rodada testa o pedido original do Igor: operar apenas as melhores entradas
relativas de cada par e deixar o breakeven/trailing determinar a expectativa.

## Elementos herdados e congelados

- mesmos ticks bid/ask, limpeza, 50 features causais e eventos da rodada 5;
- mesmos modelos por par/lado/cenário já treinados ou com configuração congelada antes dos pares
  restantes: XGBoost depth 3, learning rate 0,03, até 700 árvores e Platt em 2021-Q1;
- score `min(P_base, P_stress)`; somente o lado de maior score por timestamp pode entrar;
- treino 2019-2020, calibração 2021-Q1, seleção 2021-Q2 e validação 2021-H2, com purge/embargo de
  seis horas;
- stop inicial -1R líquido, ativação +0,5R em até 600 segundos, breakeven líquido imediato,
  trailing sem alvo a 0,5R, 1R ou 1,5R e timeout total de seis horas;
- a agenda comum usa o maior horário de saída entre base e estresse, mantendo as mesmas entradas
  sem sobreposição nos dois cenários;
- 2022, D4, S3 e semana da corretora permanecem fechados.

## Única mudança: autorização por ranking

Em cada par, depois de manter somente o lado mais forte em cada timestamp de 2021-Q2, serão testadas
as frações superiores `{0,25%; 0,5%; 1%; 2%; 5%; 10%}` do score. O corte é calculado somente em Q2
e depois congelado como valor numérico para 2021-H2; não será recalculado na validação. Cada fração
é combinada às três distâncias de trailing, totalizando 18 tentativas por par. Empates no corte são
ordenados por score, probabilidade base e timestamp, e a quantidade autorizada é exatamente
`ceil(fração × eventos)`.

## Escolha e validação

O gate Q2 continua: pelo menos 30 trades sem sobreposição, médias base e estresse positivas, dois
dos três meses positivos e profit factor de estresse acima de 1,05. O ranking das políticas que
passarem usa média de estresse, limite inferior de bootstrap e menos trades. Um par sem política
aprovada fica desligado.

Depois de congelar uma política por par, 2021-H2 é aberto uma única vez. O portfólio só pode abrir
2022 com média base e estresse positivas, bootstrap unilateral de 95% acima de zero no base, pelo
menos 300 trades, seis pares positivos, 55% dos meses positivos, concentração máxima de 35%,
PBO <= 0,20 e DSR >= 0,95 considerando as 180 tentativas. Falha encerra a rodada em 2021-H2.

Esta rodada é pesquisa offline: não ativa estratégia e não envia ordens Demo ou LIVE.
