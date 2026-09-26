# C13 frente C: descoberta (2012-01 a 2026-08, 7 moedas contra o dólar)

**C não é day trade:** as posições ficam dias ou semanas e pagam ou recebem o swap histórico.

Commit do código: a7bd11bda0d8f4b6c1e53072ea07f50741f17c37. Dias FX: 3,811. **Aprovadas: 0 de 4** (p máx-t ≤ 0,05, média > 0 com custo normal e no estresse, positiva em ≥ 60% dos anos). Retorno diário em fração do capital, com exposição bruta de até 1 (soma de |peso| ≤ 1: pernas vetadas e o aquecimento ficam em 0; a média está em mean_gross), líquido do markup de 1% ao ano por lado e do custo de C3; no estresse, markup de 2% e custo dobrado. "1x" e "3x" multiplicam essa exposição.

```
    strategy  days  mean_daily  mean_daily_stress  annual_return  annual_vol  mean_gross   sharpe  max_drawdown  positive_years       t  p_maxt  passes
       CARRY  3811  -4.509e-06         -4.309e-05      -0.001172     0.04175      0.9997 -0.02807        0.1593             0.6 -0.1075  0.9252   False
 CARRY_TREND  3811  -4.171e-05         -6.249e-05       -0.01084     0.03583      0.4938  -0.3026        0.2141             0.4  -1.159   0.999   False
       TSMOM  3811  -1.819e-05         -5.476e-05      -0.004728       0.052      0.9315 -0.09092         0.185          0.2667 -0.3482  0.9672   False
DOLLAR_CARRY  3811  -2.714e-05         -6.568e-05      -0.007056     0.06652      0.9997  -0.1061        0.2795             0.4 -0.4062  0.9746   False
```

## Em dinheiro, numa conta de US$ 1.000 (custo normal)

```
    strategy exposure  annual_usd  worst_drawdown_usd
       CARRY       1x      -1.172               159.3
       CARRY       3x      -3.516                 478
 CARRY_TREND       1x      -10.84               214.1
 CARRY_TREND       3x      -32.53               642.3
       TSMOM       1x      -4.728                 185
       TSMOM       3x      -14.18               555.1
DOLLAR_CARRY       1x      -7.056               279.5
DOLLAR_CARRY       3x      -21.17               838.5
```

## Por ano (soma dos retornos diários em %, custo normal)

```
      CARRY  CARRY_TREND  TSMOM  DOLLAR_CARRY
date                                         
2012   2.19        -0.82   0.00          1.13
2013  -4.38        -1.44  -1.42         -4.91
2014   1.38         4.30   0.27         -9.43
2015  -2.66        -5.58   7.01         -8.80
2016   1.60        -4.70  -3.19          8.81
2017  -1.58        -1.98  -3.59         -7.20
2018  -2.34         0.06  -0.99          4.87
2019   0.38        -1.79  -1.36         -0.53
2020  -1.82        -1.84   1.07         -6.46
2021   1.22        -3.04  -3.69          3.79
2022  -3.81         1.37   7.56          7.15
2023   1.29         0.23  -2.91         -0.92
2024   2.41         0.00  -0.41          8.72
2025   1.07        -0.74  -3.97         -6.25
2026   3.34         0.07  -1.33         -0.31
```

Última observação de cada juro no FRED (depois dela, o valor segue o último): USD 2026-08, EUR 2026-01, GBP 2026-01, JPY 2026-07, CHF 2026-08, AUD 2026-08, NZD 2026-08, CAD 2026-08.
