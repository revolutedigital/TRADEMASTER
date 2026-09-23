# Fast55 — diagnóstico de gates fixos conservadores

Data: 2026-09-23  
Mercado: BTCUSDT futures, `aggTrades` normalizado  
Modelo base: top-p low-cost micro congelado em Fast43  
Resultado: gate fixo melhora e fica positivo no agregado, mas ainda é frágil.

## Por que esta rodada existe

Fast54 deixou claro que o seletor walk-forward melhorou, mas ainda reagiu tarde.
Fast55 testa gates fixos conservadores, mais simples:

- `5min≤0`: só pega o primeiro sinal de cada cluster de 5 minutos;
- `15min≤5`: permite até 5 sinais recentes em 15 minutos;
- `2h≤8`: permite até 8 sinais recentes em 2 horas.

Esses gates são diagnósticos. Eles foram comparados depois da falha estendida,
então não aprovam estratégia; servem para direcionar o próximo pré-registro.

## Resultado agregado

| Gate | Sinais mantidos | Trades | Média 1 bps | Média 2 bps | Total 2 bps | Dias positivos | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5min≤0` | 427 | 427 | +1,6668 | +0,6668 | +284,7245 | 37/68 | 117,1846 |
| `15min≤5` | 1.442 | 522 | +1,4910 | +0,4910 | +256,3275 | 35/68 | 150,1286 |
| `2h≤8` | 1.005 | 325 | +1,4438 | +0,4438 | +144,2407 | 40/68 | 176,3586 |

O melhor foi `5min≤0`.

## `5min≤0` por bloco

| Bloco | Trades | Stress total | Stress médio |
| --- | ---: | ---: | ---: |
| Fast43 | 54 | +19,4083 | +0,3594 |
| Fast44 | 46 | +47,1373 | +1,0247 |
| Fast45 | 18 | +157,4910 | +8,7495 |
| Fast46 | 26 | -19,6968 | -0,7576 |
| Fast47 | 14 | +41,8213 | +2,9872 |
| Fast48 | 11 | +26,5966 | +2,4179 |
| Fast49 | 107 | -149,6414 | -1,3985 |
| Fast50 | 59 | +55,1917 | +0,9355 |
| Fast51 | 30 | -25,0023 | -0,8334 |
| Fast52 | 30 | +88,1049 | +2,9368 |
| Fast53 | 32 | +43,3139 | +1,3536 |

## Leitura

Esse é o melhor formato encontrado até agora:

- reduz o problema de excesso de exposição;
- transforma o holdout estendido de negativo para positivo no agregado;
- mantém 8/11 blocos positivos em stress.

Mas ainda é fraco para promover:

- só 37/68 dias positivos;
- t-stat diário aproximado: 1,24;
- Fast49 ainda perde -149,6414 bps;
- não usa book/quote real;
- foi escolhido depois de ver a falha estendida.

## Decisão

Ainda não temos estratégia.

Temos um próximo candidato de pesquisa:

`top-p 1% + micro_8_3 + gate 5min≤0`

O próximo passo válido é congelar esse gate e testar em dados novos ou em um
bloco pré-registrado que não participou da escolha. Sem isso, é só diagnóstico.

Sem Testnet/LIVE.

