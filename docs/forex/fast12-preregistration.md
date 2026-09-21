# Pré-registro, rodada 12: sinal + breakeven + trailing

Registrado em 2026-09-21 antes de baixar ou abrir o dia UTC 2026-09-20. Pesquisa offline;
nenhuma chave, conta ou ordem é usada.

## Pergunta

O fluxo executado não previu retorno terminal suficiente para pagar custos. Esta rodada testa a
hipótese específica de gestão de caminho: uma entrada pode identificar excursão favorável sem
prever onde o preço termina, desde que stop inicial, breakeven e trailing capturem essa excursão.

## Amostras e fronteira

- fonte: Binance Vision oficial, `BTCUSDT` spot, klines de 1 segundo;
- candidatos de entrada congelados: as 25 finalistas de cada horizonte da rodada 11;
- seleção da gestão: dias UTC 2026-09-18 e 2026-09-19;
- auditoria única e intocada: dia UTC 2026-09-20;
- o ZIP de 20/09 e seu `CHECKSUM` só podem ser baixados depois deste pré-registro e do commit da
  fábrica de gestão;
- dias 13–17 servem apenas para reconstruir causalmente os 75 sinais congelados.

## Universo de gestão

O horizonte original do sinal define o máximo de carregamento: sinal de 5 s usa 30 s, sinal de
30 s usa 120 s e sinal de 120 s usa 300 s. Para cada uma das 75 entradas são testadas sete gestões
`(stop inicial; gatilho de proteção; distância do trailing)`, todas em bps:

`(5;3;2)`, `(10;5;3)`, `(10;5;5)`, `(20;10;5)`, `(20;10;10)`, `(40;20;10)` e
`(40;20;20)`. Total: 525 hipóteses combinadas.

A entrada teórica ocorre no open do segundo seguinte. O stop é checado contra OHLC de 1 segundo com
ordem intrabar pessimista. Ao atingir o gatilho, o piso líquido vai para zero e depois acompanha a
melhor excursão menos a distância declarada. A proteção só pode ser ativada quando o movimento bruto
também cobre o custo. Stops atualizados valem no mesmo segundo; gap atravessando stop usa o pior entre
o open e o stop. Sem stop, a saída usa o close no limite de tempo.

São calculados custos roundtrip fixos de 5 bps e 20 bps. O breakeven é líquido de cada cenário, não
um falso zero antes de taxas.

## Seleção e auditoria

Cada hipótese exige ao menos 100 eventos na seleção. As 25 melhores pela média líquida de 20 bps,
com desempate pela média de 5 bps e maior amostra, são congeladas antes de abrir 20/09.

Na auditoria, exige-se ao menos 50 eventos e 24 blocos UTC distintos de 30 minutos. A correção de
Bonferroni cobre as 25 finalistas. Para passar, médias e limites inferiores unilaterais corrigidos
devem ser positivos tanto em 5 quanto em 20 bps. Zero aprovados rejeita esta arquitetura de sinal e
gestão no P0. Aprovação autoriza somente expansão offline para mais dias/ativos e top-of-book; nunca
Demo ou LIVE.
