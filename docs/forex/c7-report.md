# C7: o Claude como banca (sinais diários, grupo B, último ano)

Commit do código: 7352189dbf6ce76e4b6a7e2af2e4d083e299efe8. **Aprovado: False.** Resultado em ATR por operação, líquido do custo.

```
                          conjunto  trades  mean_atr  mean_atr_stress       t  p_one_sided  win_rate  avg_win  avg_loss
           todos os sinais diários    1083   -0.1535          -0.1795  -1.899       0.9712    0.3647    1.463    -1.082
               aceitos pelo Claude     268  -0.02513         -0.04031 -0.1294       0.5515    0.3582    2.069    -1.194
aceitos pela banca do C6 (xgboost)     154   -0.2354          -0.2644 -0.9378       0.8258    0.3312    2.352    -1.516
                aceitos pelos dois      50   -0.5663          -0.5784  -2.193       0.9858      0.32    1.328    -1.458
```

```
               sinais  aceitos
config                        
CHANDELIER_D1      33       13
NR7_D1            666      148
SQUEEZE_D1         27        6
TURTLE20          242       49
TURTLE55          115       52
```
