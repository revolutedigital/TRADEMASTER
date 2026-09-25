# C6: a banca (meta-labeling) no último ano

Commit do código: 8f78e8910a3cc4c48ce0c45f521c8e91e43caff7. Treino: 325,582 operações até 2025-08; teste: 24,016 operações de 2025-09 a 2026-08. **Aprovada: False.**

Resultado em ATR por operação, líquido do custo.

```
                                              conjunto  trades  mean_atr  mean_atr_stress       t  p_one_sided  win_rate  avg_win  avg_loss
         principal: grupo B, 12 estratégias, sem banca   17257   -0.1011           -0.212  -5.027            1    0.4986    1.144    -1.339
principal: grupo B, 12 estratégias, aceitas pela banca    1644   -0.0336         -0.08769 -0.5362       0.7041    0.5207    1.343    -1.529
   informativo: 41 mercados, 14 estratégias, sem banca   24016  -0.09561          -0.1961  -5.503            1    0.5036    1.116    -1.325
     informativo: 41 mercados, 14 estratégias, aceitas    2713  -0.01733         -0.07155 -0.3809       0.6484    0.5326    1.286    -1.502
          informativo: treino (até 2025-08), sem banca  325582  -0.06754          -0.1629   -13.1            1    0.5041    1.167    -1.322
```

Prova principal por estratégia (aceitas / todas):

```
               count    mean  total  mean_all
config                                       
CHANDELIER_D1      1   1.005     33    -0.342
CHANDELIER_H4    106 -0.3953    290   -0.5769
MACD200_H1       233 -0.2727   2039   -0.2824
MACD200_H4        51  0.2049    533  -0.09382
NR7_D1            33 0.01297    666  -0.05288
NR7_H4           173 0.03367   3612  -0.08892
RSI2_H1          627 0.09992   7659  -0.04552
RSI2_H4          271  0.0534   1871   -0.0421
SQUEEZE_H4        29 -0.2393    170   -0.2137
TURTLE20          87 -0.3814    242   -0.3483
TURTLE55          33 -0.1367    115   -0.2805
```

Peso de cada avaliador no modelo:

```
rsi14           0.09197
trend50         0.07951
hour            0.07583
cost_atr        0.07558
trend200        0.07094
trend200_side   0.06237
vol_h1          0.05448
config_id       0.05404
slope50         0.05258
vol_d1          0.05023
trend50_side    0.04765
recent_config   0.04664
class_id         0.0451
slope50_side    0.04415
side            0.03871
weekday         0.03791
ret4h_side      0.03718
recent_same     0.03512
```
