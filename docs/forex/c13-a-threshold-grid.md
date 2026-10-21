# C13-A: teste de thresholds diferentes

Teste offline, sem tocar no robô demo.

Thresholds testados:

```
0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.10, 0.12, 0.16
```

Gestões comparadas:

1. `base: stop 1 ATR · alvo 3R`;
2. `base: stop 0.5 ATR · alvo 5R`;
3. `parcial literal: stop 1 ATR · 50% em 0.5R + resto 3R + trail 1R`;
4. `parcial livre: stop 1 ATR · 50% em 0.5R + resto sem alvo + trail 1R`.

Arquivos:

- `data/c13/partial_threshold_stats.parquet`;
- `data/c13/partial_threshold_money.parquet`.

## Leitura principal

O threshold `0.05` continua sendo o melhor centro de decisão.

O `0.04` pareceu ótimo em 2016-2022, mas várias combinações explodiram risco fora da escolha em 2023-2026. Então ele pode ser estudado, mas não é candidato seguro sem filtro adicional de concentração/risco.

O `0.02` coloca trades demais. Ele aumenta operações e às vezes lucro bruto, mas joga mais exposição para sinais fracos e tende a piorar risco/concentração.

Acima de `0.08`, a qualidade por trade sobe, mas o volume cai demais. É bom para confirmar que o modelo ordena bem, mas não parece ótimo para dinheiro anual.

## Candidatas que respeitam risco até 25% em 2016-2022 e 2023-2026

Ordenadas por lucro em stress no período fora da escolha (`2023-2026`):

```
threshold  gestão                                                         risco  teto  stress 16-22  risco 16-22  stress 23-26  risco 23-26
0.05       base: stop 1 ATR · alvo 3R                                    2%     5%    US$ 440.6     24.74%       US$ 808.9     23.62%
0.05       parcial livre: 50% em 0.5R + resto sem alvo + trail 1R        2%     5%    US$ 385.3     20.40%       US$ 763.8     19.53%
0.02       parcial literal: 50% em 0.5R + resto 3R + trail 1R            1%     5%    US$ 294.8     22.34%       US$ 722.8     22.92%
0.05       parcial literal: 50% em 0.5R + resto 3R + trail 1R            2%     5%    US$ 363.4     19.93%       US$ 712.0     21.82%
0.05       base: stop 1 ATR · alvo 3R                                    1%     5%    US$ 406.5     17.80%       US$ 685.8     16.40%
0.04       base: stop 0.5 ATR · alvo 5R                                  0.5%   2%    US$ 403.0     11.63%       US$ 624.6     21.51%
0.05       base: stop 0.5 ATR · alvo 5R                                  0.5%   2%    US$ 315.2     11.68%       US$ 602.9     13.05%
0.02       base: stop 0.5 ATR · alvo 5R                                  0.5%   2%    US$ 248.4     15.20%       US$ 602.2     15.44%
0.03       base: stop 0.5 ATR · alvo 5R                                  0.5%   2%    US$ 303.1     17.10%       US$ 577.6     16.53%
```

## Melhor risco × retorno robusto

Se a régua for eficiência e não lucro bruto:

```
threshold 0.05
base: stop 0.5 ATR · alvo 5R
risco 0.5%
teto aberto 2%
```

Fez:

- `2016-2022`: `US$ 315.2/ano` em stress, risco `11.68%`;
- `2023-2026`: `US$ 602.9/ano` em stress, risco `13.05%`.

## Melhor lucro mantendo risco até 25%

```
threshold 0.05
base: stop 1 ATR · alvo 3R
risco 2%
teto aberto 5%
```

Fez:

- `2016-2022`: `US$ 440.6/ano` em stress, risco `24.74%`;
- `2023-2026`: `US$ 808.9/ano` em stress, risco `23.62%`.

## Melhor versão da ideia parcial/trailing

```
threshold 0.05
stop 1 ATR
sai 50% em 0.5R
resto sem alvo fixo
trailing 1R
risco 2%
teto aberto 5%
```

Fez:

- `2016-2022`: `US$ 385.3/ano` em stress, risco `20.40%`;
- `2023-2026`: `US$ 763.8/ano` em stress, risco `19.53%`.

Essa é a melhor se quisermos um robô mais confortável, com parcial e trailing. Ela perde para a base em lucro máximo, mas entrega um perfil psicológico e de risco melhor.

## O que não escolher agora

Evitar escolher só porque ficou bonito no treino:

- `threshold 0.04 · stop 1 ATR · alvo 3R · risco 1% · teto 10%` fez `US$ 890.6/ano` em 2016-2022, mas leu risco `42.32%` em 2023-2026.
- `threshold 0.04 · parcial livre · risco 2% · teto 10%` fez `US$ 824.9/ano` em 2016-2022, mas leu risco `80.18%` em 2023-2026.

Ou seja: `0.04` pode esconder concentração demais. Só entra como candidato se adicionarmos filtro de concentração/risco aberto mais duro.

## Decisão técnica provisória

Para bot demo/monitoramento, as três linhas candidatas são:

1. conservadora eficiente: `0.05 · stop 0.5 ATR · alvo 5R · risco 0.5% · teto 2%`;
2. agressiva controlada: `0.05 · stop 1 ATR · alvo 3R · risco 2% · teto 5%`;
3. confortável/trailing: `0.05 · parcial livre · risco 2% · teto 5%`.

O threshold que sobreviveu melhor foi `0.05`.
