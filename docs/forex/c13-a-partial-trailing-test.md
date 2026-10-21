# C13-A: teste da ideia “mão mais pesada, 50% fora cedo, resto no zero a zero + trailing”

Hipótese do Igor:

> entrar com mão mais pesada, sair 50% em `0,5 ATR`, colocar o resto no zero a zero, buscar `3R` e carregar no trailing até stopar com lucro.

Teste offline, sem tocar no robô demo:

- threshold fixo: `0.05`;
- candidatas: as mesmas 36.735 trades aceitas pelo modelo C13-A em 2016-2026;
- comparação contra as duas bases fortes já conhecidas:
  - `stop 1 ATR · alvo 3R`;
  - `stop 0.5 ATR · alvo 5R`;
- política de dinheiro: sem juros compostos, `volume_minimo=pula`;
- riscos testados: `0,5%`, `1%`, `2%`, `3%`, `5%`;
- tetos de risco aberto: `2%`, `5%`, `10%`, `15%`.

Arquivos de resultado:

- `data/c13/partial_trailing_stats.parquet`;
- `data/c13/partial_trailing_money.parquet`.

## Resultado em R

No período inteiro `2016-2026`, a parcial/trailing aumenta acerto, mas corta payoff. Ela fica mais suave, porém não melhora o edge bruto.

```
gestão                                                       média_R  stress_R  acerto  t
base: stop 0.5 ATR · alvo 5R                                 0.1396   0.1007    48.95%  6.091
base: stop 1 ATR · alvo 3R                                   0.0827   0.0633    54.31%  5.661
stop 0.5 ATR · 50% em 1R/0.5ATR + resto 3R + trail 1R         0.0962   0.0573    56.15%  5.527
stop 1 ATR · 50% em 0.5R + resto 3R + trail 1R                0.0540   0.0345    61.31%  4.881
```

Leitura: a ideia funciona no sentido de aumentar taxa de acerto, mas a saída parcial cedo tira exatamente o que pagava a estratégia: os trades grandes.

## Resultado em dinheiro / risco

Ranking por eficiência até `25%` de risco de referência, escolhendo em `2016-2022` e lendo em `2023-2026`:

```
gestão                                                        risco  teto  16-22 stress/ano  16-22 risco  23-26 stress/ano  23-26 risco
base: stop 0.5 ATR · alvo 5R                                  0.5%  2%    US$ 315.2         11.68%       US$ 602.9         13.05%
base: stop 1 ATR · alvo 3R                                    1.0%  5%    US$ 406.5         17.80%       US$ 685.8         16.40%
base: stop 1 ATR · alvo 3R                                    1.0%  2%    US$ 193.7          8.55%       US$ 304.7         11.09%
stop 1 ATR · 50% em 0.5R + resto sem alvo + trail 1R          1.0%  2%    US$ 152.6          7.33%       US$ 288.4         10.75%
stop 1 ATR · 50% em 0.5R + resto sem alvo + trail 1R          2.0%  5%    US$ 385.3         20.40%       US$ 763.8         19.53%
stop 1 ATR · 50% em 0.5R + resto 3R + trail 1R                2.0%  5%    US$ 363.4         19.93%       US$ 712.0         21.82%
base: stop 1 ATR · alvo 3R                                    2.0%  5%    US$ 440.6         24.74%       US$ 808.9         23.62%
```

## Conclusão

A ideia é boa como controle psicológico/risco, mas não como melhor edge matemático.

O que ela faz bem:

- aumenta a taxa de acerto;
- reduz a sensação de “ganhou e devolveu tudo”;
- permite usar `2%` por trade com teto `5%` mantendo risco de referência perto de `20%`.

O que ela faz mal:

- corta os trades que pagam a conta;
- perde para `stop 0.5 ATR · alvo 5R` em eficiência risco/retorno;
- perde para `stop 1 ATR · alvo 3R` em lucro absoluto sob risco até `25%`.

Se quisermos seguir com essa ideia, a melhor versão não é exatamente “resto alvo 3R”. A melhor leitura foi:

```
threshold 0.05
stop 1 ATR
sai 50% em 0.5R
resto sem alvo fixo
stop do resto no zero a zero
trailing 1R
risco 2%
teto aberto 5%
```

Ela leu `US$ 763,8/ano` em stress em 2023-2026 com risco de referência `19,53%`.

Mas se a régua for retorno matemático puro, a candidata preferida continua:

```
threshold 0.05
stop 0.5 ATR
alvo 5R
risco 0.5%
teto aberto 2%
```

E, para mão mais pesada controlada:

```
threshold 0.05
stop 1 ATR
alvo 3R
risco 2%
teto aberto 5%
```
