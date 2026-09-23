# Fast41–45 — candidato low-cost com top-p e micro-trailing congelado

Data: 2026-09-23  
Mercado: BTCUSDT futures, `aggTrades` normalizado  
Período de treino/calibração: 2026-06-30 a 2026-07-06  
Holdouts congelados: 2026-07-07 a 2026-07-27  
Resultado: candidato encontrado, ainda research-only.

## O que mudou

As rodadas anteriores rejeitaram dois caminhos:

- custos altos de spot/market taker destruíam qualquer lift;
- terminal return/top-p não bastava para virar edge executável;
- nos WALs com book real, o edge era forte em algumas janelas e quebrava em
  outras.

Fast41–45 testou uma hipótese mais estreita:

> Se o custo roundtrip real ficar perto de 1–2 bps, um modelo top-p consegue
> identificar entradas que andam o suficiente para um trailing micro carregar?

Para responder sem misturar premissas, o builder passou a aceitar custo/horizonte
via CLI e o replay passou a usar os custos do manifest em vez de ficar travado em
12/24 bps.

## Configuração

Dataset low-cost:

- decisões a cada 5 s;
- horizontes: 30, 120 e 300 s;
- custo esperado: 1 bps roundtrip;
- custo stress: 2 bps roundtrip;
- latência simulada: 100 ms;
- stop inicial de label: 20 bps;
- uma posição por vez no replay de portfólio.

Políticas micro adicionadas:

| Política | Stop inicial | BE ativa | BE lock | Trailing ativa | Distância |
| --- | ---: | ---: | ---: | ---: | ---: |
| `micro_2_1` | 6 bps | 2 bps | 2 bps | 3 bps | 1 bps |
| `micro_3_1` | 8 bps | 3 bps | 2 bps | 5 bps | 1 bps |
| `micro_5_2` | 12 bps | 5 bps | 2 bps | 8 bps | 2 bps |
| `micro_8_3` | 16 bps | 8 bps | 2 bps | 12 bps | 3 bps |

## Fast41 — payoff terminal low-cost

Fonte:

`backend/data/microstructure_v1/reports/direct-payoff-scout-fast41-lowcost-jun30-jul06-100ms/report.json`

Hash:

`eaa0921b28d3e7a16069751a046a31dd4d44eea9bc0df521341f1d49d8a2222f`

Resultado:

- melhor candidato: horizonte 120 s, `flow_price`, top 1%;
- top 1% realizou `-1,4989 bps` já com stress de 2 bps;
- melhor replay ficou `-0,5013 bps` em custo 1 bps e `-1,5013 bps` em custo
  2 bps.

Decisão: payoff terminal segue rejeitado.

## Fast42 — probabilidade de andar + micro-trailing

Fonte:

`backend/data/microstructure_v1/reports/latency-adjusted-scout-fast42-lowcost-micro-jun30-jul06-100ms/report.json`

Hash:

`a3011b50ee61036c03f0245ebc3c4d6750100fb2ac9f2e7a4a56631e5a58b2f3`

Melhor candidato de desenvolvimento:

- alvo: `latency_paid_stress_before_stop`;
- horizonte: 300 s;
- features: `flow_price_session`;
- top-p: 1%;
- política: `micro_8_3`;
- sinais selecionados: 4.303;
- target rate top 1%: 87,06%;
- AUC média: 0,5693.

Replay de desenvolvimento:

| Política | Top-p | Trades | Custo 1 bps | Custo 2 bps | Observação |
| --- | ---: | ---: | ---: | ---: | --- |
| `micro_8_3` | 1% | 287 | +0,5565 bps | -0,4435 bps | positivo só no custo esperado |
| `micro_8_3` | 2% | 367 | +0,4181 bps | -0,5819 bps | stress negativo |
| `micro_8_3` | 5% | 502 | +0,3622 bps | -0,6378 bps | stress negativo |
| `micro_5_2` | 5% | 778 | +0,3349 bps | -0,6651 bps | stress negativo |

Decisão: não aprova estratégia, mas gera candidato para holdout congelado.

## Candidato congelado

Treinado/calibrado somente em 2026-06-30 a 2026-07-06.

- model hash: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- calibração: 2026-07-06;
- horizonte: 300 s;
- features: `flow_price_session`;
- alvo: `latency_paid_stress_before_stop`;
- top-p: 1%;
- threshold congelado: `0,90195311025804`;
- política: `micro_8_3`;
- custo esperado/stress: 1/2 bps;
- autorização: nenhuma.

## Fast43–45 — holdouts congelados

| Rodada | Holdout | Sinais top-p | Target rate selecionado | Trades | Dias com trade | Média 1 bps | Média 2 bps | Total 2 bps | Max DD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast43 | 2026-07-07 a 2026-07-13 | 432 | 93,06% | 100 | 6 | +2,0319 | +1,0319 | +103,1949 | 83,2690 |
| Fast44 | 2026-07-14 a 2026-07-20 | 369 | 92,95% | 89 | 6 | +3,1991 | +2,1991 | +195,7242 | 54,2140 |
| Fast45 | 2026-07-21 a 2026-07-27 | 120 | 95,00% | 25 | 6 | +8,2941 | +7,2941 | +182,3520 | 18,6419 |
| Consolidado | 2026-07-07 a 2026-07-27 | 921 | — | 214 | 18 | +3,2489 | +2,2489 | +481,2711 | 83,2690 |

Saídas consolidadas:

- `BREAKEVEN_STOP`: 147;
- `INITIAL_STOP`: 42;
- `TIMEOUT`: 13;
- `TRAILING_STOP`: 12.

Distribuição diária no stress de 2 bps:

- dias positivos: 13/18;
- pior dia: -59,1212 bps;
- melhor dia: +132,6866 bps;
- média de total diário: +26,7373 bps;
- t-stat aproximado por total diário: 2,34.

Hashes:

| Rodada | Report | Trades |
| --- | --- | --- |
| Fast43 | `97acfa6a4472fdc5083dea08af4d6bd26c0566cf2d145f73d352466b63c5fdaa` | `fabc84ac345d1f1b5f97604a293691291932983bb1f45a8f3cf898648854f42d` |
| Fast44 | `d81fa62232a5de6b5a8de8b7a493de7b6700f4592e1a9568d5e6e9ac0276273f` | `695f2512c73dfc1c8eae96aa3c80c36b374a845eb830469c1ec4e33fa6b7fdd5` |
| Fast45 | `2de2eebcf27103652f364420c7a4a810d863080061e67d104aa8d3b425e857fe` | `5f695e43893a76690a5198a325d70c5e42f7b73022a4372007cf2298056d7d92` |

## Decisão

Temos um candidato real.

Ainda não temos estratégia aprovada.

Motivo para continuar:

- o modelo e threshold foram congelados antes dos três holdouts;
- todos os três blocos ficaram positivos no stress de 2 bps;
- o resultado consolidado pagou custo esperado e stress;
- o top-p selecionado manteve target rate muito acima do base rate.

Motivo para não promover ainda:

- o teste usa `aggTrades`, não top-of-book bid/ask real;
- custo 1–2 bps é premissa, não medição de execução na nossa conta/venue;
- não modela fila, parcial, liquidez, gap e rejeição;
- é um único ativo e uma janela de julho;
- Fast45 teve só 25 trades, então a queda de frequência precisa ser entendida;
- ainda falta atravessar agosto/setembro e um replay com book/quote ou shadow
  prospectivo.

Próximo gate:

1. aplicar o mesmo artefato congelado em todos os blocos semanais restantes até
   2026-09-20;
2. exigir positividade em stress por bloco e por mês;
3. repetir com replay top-of-book quando houver book/quote suficiente;
4. só depois considerar shadow/Testnet research-only;
5. LIVE continua fora de escopo.

Sem Testnet/LIVE.

## Atualização pós-gate

O próximo gate foi executado em Fast46–53. O candidato não sobreviveu ao período
completo até 2026-09-20.

Ver:

`docs/forex/fast53-lowcost-micro-extended-holdout-report.md`
