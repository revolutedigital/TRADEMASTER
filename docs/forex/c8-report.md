# C8: a banca de saída (último ano, 37 mercados)

Commit do código: ab5cfcc0a37da209ac1bb247193c98af37974e87. **Banca de saída aprovada: False. Entrada + saída aprovadas: False.** Treino: 259,283 pontos de decisão; teste: 2,341 operações, 1,133 realizadas antes pela banca. Resultado em ATR por operação, líquido do custo.

```
                         conjunto  trades  mean_atr  mean_atr_stress       t  p_one_sided  win_rate  avg_win  avg_loss
                   saída original    2341   -0.2483          -0.2806   -4.76            1    0.3806    1.247    -1.167
             com a banca de saída    2341  -0.07614          -0.1085  -2.386       0.9915    0.5664   0.7154     -1.11
             melhora por operação    2341    0.1722           0.1722   4.497     3.44e-06    0.2922    1.362   -0.3192
            só a banca de entrada     522   -0.1785           -0.215  -1.721       0.9574    0.4234    1.462    -1.383
banca de entrada + banca de saída     522  -0.02074         -0.05718 -0.2996       0.6177    0.6149   0.8154    -1.356
```

```
               operacoes  original  com_banca  realizadas_antes
config                                                         
CHANDELIER_D1         41   -0.5915    -0.2495                30
CHANDELIER_H4        372   -0.5638    -0.2481               191
NR7_D1               808  -0.08361   -0.04735               333
RSI2_D1              391    0.1137     0.1351               121
SQUEEZE_D1            37   -0.3324   -0.08802                25
TSMOM                224   0.07081   -0.03558               111
TURTLE20             315   -0.5974   -0.09111               221
TURTLE55             153   -0.9119     -0.329               101
```
