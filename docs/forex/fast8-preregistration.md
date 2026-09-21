# Pré-registro, rodada 8: time stop antes do stop cheio

Registrado em 2026-09-21 após a reprovação da rodada 7 e antes de simular qualquer novo timeout.
H2 e todos os períodos posteriores continuam fechados.

## Evidência que gerou a hipótese

Com Top 5%, proteção +0,1R e trailing 0,5R, a rodada 7 elevou a ativação, mas continuou negativa.
Nos stops stress, o primeiro quartil ocorreu em 51,8 segundos, a mediana em 105,8 e o terceiro
quartil em 207,5 segundos. Um trade que não responde nesse intervalo pode ser encerrado antes de
consumir -1R.

## Política congelada

- mesmos modelos, calibradores, `trusted_score`, Top P causal global e piso absoluto da rodada 6;
- Top P 5%, breakeven em +0,1R e trailing 0,5R;
- stop inicial -1R, timeout total seis horas, custos base/estresse;
- três posições globais, uma por par, mesma agenda comum entre cenários;
- somente o timeout pré-ativação muda.

## Tentativas e gate

Timeouts `{60; 120; 210}` segundos, escolhidos antes dos resultados e próximos aos quartis
observados. São três novas tentativas, levando o total acumulado desta linha de pesquisa a 24
(18 da rodada 6, 3 da rodada 7 e 3 desta rodada).

O gate Q2 permanece: 100 trades, médias base e estresse positivas, dois dos três meses positivos,
profit factor stress acima de 1,05, quatro pares participantes e concentração máxima de 35%. A
melhor política usa média stress, bootstrap e menos trades. Se nenhuma passar, esta arquitetura de
entrada probabilística + breakeven + trailing é encerrada com o conjunto atual de features.

Somente aprovação Q2 permite uma abertura única de 2021-H2. 2022, D4, S3, corretora, Demo e LIVE
continuam fechados.
