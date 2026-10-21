# C13-A: ranking de risco × retorno do sweep de threshold, stop e alvo

Fonte: `docs/forex/c13-a-sweep.md`, `data/c13/sweep_stats.parquet` e `data/c13/sweep_money.parquet`.

Este ranking usa o teste já rodado do C13-A:

- thresholds: `0.02`, `0.05`, `0.08`, `0.12`, `0.16`;
- 40 gestões: stops `0.5`, `1`, `1.5`, `2 ATR`, alvos `1R`, `2R`, `3R`, `5R`, saídas parciais e trailing;
- simulação financeira a partir de US$ 1.000, sem juros compostos;
- política conservadora: `volume_minimo=pula` — se o lote mínimo arrisca mais que o orçamento, pula a operação;
- escolha feita em `2016-2022`, leitura fora da escolha em `2023-2026`;
- risco de referência = `max(pior_queda, pior_queda_todos_stops)`;
- eficiência = `lucro_ano_estresse_usd / risco_de_referencia`.

Isto ainda é pesquisa. Não autoriza alterar robô demo/live sozinho.

## Conclusão curta

O limiar `0.05` domina as escolhas robustas. `0.02` gera mais trades e às vezes mais lucro bruto, mas piora concentração/risco. Acima de `0.08`, a qualidade por trade sobe, mas o volume cai e o lucro anual perde força.

Para risco × retorno conservador, a melhor candidata do sweep é:

```
threshold 0.05
gestão: stop 0.5 ATR · alvo 5R
risco por trade: 0.5%
teto aberto: 2%
volume mínimo: pula
```

Ela escolhe em 2016-2022:

- stress: `US$ 315,2/ano`;
- risco ref.: `11,68%`;
- eficiência: `2.699`;

e lê em 2023-2026:

- stress: `US$ 602,9/ano`;
- risco ref.: `13,05%`;
- operações: `3.424`.

Se o objetivo for mais lucro absoluto mantendo risco controlado, a melhor candidata prática é:

```
threshold 0.05
gestão: stop 1 ATR · alvo 3R
risco por trade: 1%
teto aberto: 5%
volume mínimo: pula
```

Ela escolhe em 2016-2022:

- stress: `US$ 406,5/ano`;
- risco ref.: `17,8%`;
- eficiência: `2.284`;

e lê em 2023-2026:

- stress: `US$ 685,8/ano`;
- risco ref.: `16,4%`;
- operações: `3.737`.

Para aceitar até ~25% de risco, a versão mais forte é:

```
threshold 0.05
gestão: stop 1 ATR · alvo 3R
risco por trade: 2%
teto aberto: 5%
volume mínimo: pula
```

Ela escolhe em 2016-2022:

- stress: `US$ 440,6/ano`;
- risco ref.: `24,74%`;

e lê em 2023-2026:

- stress: `US$ 808,9/ano`;
- risco ref.: `23,62%`;

## Top por eficiência sob risco ref. ≤ 20%

```
limiar  gestão                                      risco  teto  ops_16_22  stress_16_22  risco_ref  eficiência  stress_23_26  risco_23_26  ops_23_26
0.05    stop 0.5 ATR · alvo 5R                    0.005  0.02  6219       315.2         0.1168     2699        602.9         0.1305       3424
0.05    stop 1 ATR · empate em 1R + trailing 1R   0.01   0.02  3153       197.6         0.0805     2454        311.5         0.1264       1700
0.05    stop 1 ATR · metade em 1R + trailing 1R   0.01   0.02  3153       190.7         0.0835     2285        306.8         0.1159       1700
0.05    stop 1 ATR · alvo 3R                      0.01   0.05  6777       406.5         0.1780     2284        685.8         0.1640       3737
0.05    stop 1 ATR · alvo 3R                      0.01   0.02  3119       193.7         0.0855     2266        304.7         0.1109       1677
```

## Top por lucro absoluto sob risco ref. ≤ 20%

```
limiar  gestão                                      risco  teto  stress_16_22  risco_ref  eficiência  stress_23_26  risco_23_26
0.05    stop 1 ATR · alvo 3R                      0.01   0.05  406.5         0.1780     2284        685.8         0.1640
0.05    stop 1 ATR · alvo 2R                      0.01   0.05  388.1         0.1813     2140        664.6         0.1640
0.05    stop 2 ATR · metade em 1R + trailing 1R   0.02   0.10  386.2         0.1920     2011        603.5         0.2064
0.05    stop 1 ATR · empate em 1R + trailing 1R   0.01   0.05  385.3         0.1714     2248        665.6         0.1908
0.05    stop 2 ATR · empate em 1R + trailing 1R   0.02   0.10  383.6         0.1926     1992        598.3         0.2064
```

## Top por lucro absoluto sob risco ref. ≤ 25%

```
limiar  gestão                                      risco  teto  stress_16_22  risco_ref  eficiência  stress_23_26  risco_23_26
0.05    stop 1 ATR · alvo 3R                      0.02   0.05  440.6         0.2474     1781        808.9         0.2362
0.05    stop 1 ATR · metade em 1R + resto em 3R   0.02   0.05  437.5         0.2440     1793        760.1         0.2361
0.05    stop 1 ATR · alvo 3R                      0.01   0.05  406.5         0.1780     2284        685.8         0.1640
0.02    stop 1 ATR · alvo 3R                      0.01   0.05  399.1         0.2157     1850        685.3         0.2990
0.05    stop 1 ATR · alvo 2R                      0.01   0.05  388.1         0.1813     2140        664.6         0.1640
```

## Cuidado com a faixa agressiva

A combinação com maior lucro sob teto 35%-50% em 2016-2022 foi:

```
threshold 0.05
stop 1 ATR · empate em 1R + trailing 1R
risco 2%
teto aberto 10%
```

Ela fez `US$ 945,9/ano` em stress em 2016-2022 e `US$ 1.293/ano` em 2023-2026, mas a leitura de 2023-2026 chegou a `49,77%` de risco ref. Portanto ela não é uma boa candidata se a regra real for segurar drawdown perto de 20%-25%.

## Próximo teste correto

Levar para monitoramento/runner demo apenas 2 ou 3 finalistas, sem mudar autorização live:

1. eficiência conservadora: `threshold 0.05 · stop 0.5 ATR · alvo 5R · risco 0.5% · teto 2%`;
2. prática equilibrada: `threshold 0.05 · stop 1 ATR · alvo 3R · risco 1% · teto 5%`;
3. prática agressiva controlada: `threshold 0.05 · stop 1 ATR · alvo 3R · risco 2% · teto 5%`.
