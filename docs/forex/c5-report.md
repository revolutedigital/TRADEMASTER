# C5: TSMOM e RSI2 em índices de ações

Commit do código: d62700996b71d0e0b2c7265186e20809fdadb8f7. Aprovadas nas duas amostras: nenhuma.

## Amostra X (índices já vistos, 2000-2011)

```
 config  trades  mean_atr  mean_atr_stress     t  p_holm  win_rate  avg_win  avg_loss  indices_positive  approved
  TSMOM     279     1.027           0.5055 1.975 0.03662    0.3369    5.719    -1.358                 1      True
RSI2_D1     713    0.1273          0.06259  2.09 0.03662    0.6452   0.8932    -1.265                 1      True
```

Qualidade:

```
symbol  days  years  flat_open_share  included
 ^GSPC  3019  11.99           0.5398     False
  ^NDX  3019  11.99           0.0142      True
^GDAXI  3054  11.99           0.0007      True
 ^N225  2944  11.99           0.0000      True
 ^FCHI  3067  11.99           0.0003      True
 ^AXJO  3031  11.99           0.1380      True
  ^HSI  2963  11.99           0.0014      True
 ^FTSE  3032  11.99           0.9594     False
```

Por índice (operações / média em ATR):

```
              TSMOM       RSI2_D1
symbol                           
^AXJO   68 / +1.093  109 / +0.042
^FCHI   50 / +0.893  114 / +0.293
^GDAXI  23 / +1.837  118 / +0.025
^HSI    31 / +2.138  135 / +0.248
^N225   51 / +0.791  121 / +0.034
^NDX    56 / +0.331  116 / +0.105
```

## Amostra Y (índices novos, 2000-2026)

```
 config  trades  mean_atr  mean_atr_stress      t  p_holm  win_rate  avg_win  avg_loss  indices_positive  approved
  TSMOM    1168    0.3913         -0.07002  1.405  0.1601    0.3253    4.455    -1.568              0.75     False
RSI2_D1    2249   0.02999         -0.05069 0.7446  0.2283    0.6203   0.9149    -1.416              0.75     False
```

Qualidade:

```
 symbol  days  years  flat_open_share  included
   ^RUT  6705  26.66           0.2299     False
  ^IBEX  6782  26.66           0.0544      True
  ^SSMI  6692  26.66           0.0033      True
   ^AEX  6817  26.66           0.0022      True
^GSPTSE  6697  26.66           0.0335      True
  ^KS11  6565  26.66           0.0009      True
  ^TWII  6537  26.66           0.0015      True
  ^BVSP  6607  26.66           0.2357     False
   ^MXX  6693  26.66           0.0932      True
   ^STI  6674  26.66           0.1696      True
```

Por índice (operações / média em ATR):

```
                TSMOM       RSI2_D1
symbol                             
^AEX     128 / +0.465  287 / +0.085
^GSPTSE  127 / +0.486  287 / +0.151
^IBEX    112 / +1.144  282 / +0.060
^KS11    176 / +0.664  277 / +0.057
^MXX     188 / +0.338  285 / -0.042
^SSMI    183 / -0.030  286 / +0.169
^STI     130 / -0.086  258 / -0.373
^TWII    124 / +0.355  287 / +0.094
```
