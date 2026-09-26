# C13, frente A: banca de day trade, descoberta (walk-forward 2016 a 2026-08)

Commit do código: bd03134ec9f6fc991e8e6387a8698aa65103a7b2. Candidatas: 3,252,422 operações de 34 conjuntos em 37 mercados (2012-01 a 2026-08), todas com a gestão stop 1 ATR · alvo 3R · até o fim do dia. Unidade: R líquido do custo; 1 R = US$ 5 com 0,5% de risco numa conta de US$ 1.000.

**Escolhida: profundidade 3, peso mínimo por folha 2000, τ = 0.05. Passa para o período intocado: sim.**

## As 8 combinações (operações aceitas de 2016 a 2026-08, cada ano decidido por um modelo treinado só com operações que saíram antes dele)

```
 max_depth  min_child_weight  threshold  trades  mean_net  mean_stress  win_rate     t  p_one_sided  market_share  markets  trades_per_day
         3               200          0  570977   0.02231     0.003373    0.5138 6.562     2.66e-11        0.8378       37           205.8
         3               200       0.05   36013   0.07624       0.0565    0.5407 5.303    5.705e-08        0.8378       37           12.98
         3              2000          0  574728   0.02255     0.003588     0.514 6.689    1.121e-11        0.8378       37           207.2
         3              2000       0.05   36735   0.08271      0.06327    0.5431 5.661    7.537e-09        0.8108       37           13.24
         5               200          0  675146   0.01983    0.0003576    0.5131 6.473    4.799e-11        0.8378       37           243.4
         5               200       0.05   79551   0.05708       0.0369    0.5293 5.783    3.663e-09        0.8649       37           28.68
         5              2000          0  673213   0.01993    0.0004967    0.5127 6.509    3.793e-11        0.8108       37           242.7
         5              2000       0.05   76275   0.06496      0.04458    0.5351 6.443    5.865e-11        0.9189       37            27.5
```

Informativo, todas as candidatas do mesmo período, sem banca: {"trades": 2385601, "mean_net": -0.019798321189486472, "mean_stress": -0.04267355975146402, "win_rate": 0.47604649729774595, "t": -17.07360227941437, "p_one_sided": 1.0, "market_share": 0.02702702702702703, "markets": 37, "trades_per_day": 859.9859408795962}

## A escolhida, por ano

```
      trades  mean_net
year                  
2016    4839  -0.01034
2017    3123   0.04742
2018    2857   0.03798
2019    3120    0.1688
2020    4372   0.06185
2021    2606    0.0213
2022    3160    0.1175
2023    2427   0.03668
2024    3218    0.1309
2025    3894   0.06926
2026    3119    0.2655
```

Em dinheiro (0,5% de risco por operação, US$ 1.000): 5.0

Modelo final (operações que saíram até 2026-08-31), sha256: 85e6bf5f434c8fe682f331fe33121d25ea3b14c2c0cd804251d0691a0813837d.
