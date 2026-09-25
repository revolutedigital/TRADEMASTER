# C11: descoberta (2012-01 a 2026-08)

Commit do código: 8b17d80e2074fb41e0c09db35bc916f0005c9f65. **Aprovadas: 0 de 11.** Unidade: R = 1 ATR diário (o stop de proteção); saída no fim da janela.

```
 config                  window  trades  win_rate    mean_r  mean_r_stress       t  p_one_sided  markets  market_share  p_maxt  approved
      0        fix_tokyo_before   27886    0.4765 -0.005184       -0.01866  -3.921            1        8          0.25       1     False
      1         fix_tokyo_after   29700    0.4562   -0.0109       -0.02815  -7.979            1        8         0.125       1     False
      2          fix_ecb_before   29904    0.4502  -0.01408       -0.03317  -8.804            1        8         0.125       1     False
      3           fix_ecb_after   29936    0.4455  -0.01895       -0.03806  -6.543            1        8             0       1     False
      4       fix_london_before   29918    0.4662  -0.01895       -0.03804  -7.405            1        8             0       1     False
      5        fix_london_after   29896     0.434  -0.01925       -0.03835  -10.66            1        8             0       1     False
      6 month_end_london_before    1367    0.4521  -0.02908       -0.04842  -1.783       0.9627        8         0.125       1     False
      7  month_end_london_after    1371    0.5208  0.004244       -0.01509  0.4314       0.3331        8          0.75  0.9866     False
      8            overnight_us    7257    0.5275 -0.001179       -0.02893 -0.1996       0.5791        2             0  0.9996     False
      9          europe_open_us    7481    0.4332  -0.02133        -0.0491  -8.244            1        2             0       1     False
     10           turn_of_month    5220    0.4979  -0.01089       -0.03582 -0.5618       0.7129        8         0.375       1     False
```

Por ano:

```
year                      2012   2013   2014   2015   2016   2017   2018   2019   2020   2021   2022   2023   2024   2025   2026
fix_tokyo_before         0.001 -0.002 -0.010 -0.011 -0.006 -0.002 -0.014 -0.009  0.002  0.009  0.011 -0.013 -0.007 -0.009 -0.024
fix_tokyo_after         -0.000 -0.007 -0.008 -0.010 -0.006 -0.005 -0.011 -0.018 -0.003 -0.014 -0.008 -0.016 -0.021 -0.015 -0.029
fix_ecb_before          -0.001 -0.000  0.001 -0.001  0.002 -0.017 -0.014 -0.024 -0.031 -0.026 -0.009 -0.023 -0.025 -0.022 -0.029
fix_ecb_after           -0.010 -0.029 -0.004 -0.014 -0.012 -0.014 -0.016 -0.031 -0.023 -0.036 -0.019 -0.028 -0.027 -0.006 -0.018
fix_london_before       -0.010 -0.019 -0.022  0.004 -0.005 -0.035 -0.015 -0.038 -0.015 -0.016 -0.012 -0.026 -0.017 -0.040 -0.020
fix_london_after        -0.007 -0.011 -0.019 -0.001 -0.005 -0.019 -0.024 -0.032 -0.023 -0.024 -0.014 -0.013 -0.038 -0.036 -0.025
month_end_london_before -0.056 -0.016  0.044 -0.131 -0.119  0.011  0.012  0.044 -0.015 -0.232  0.076 -0.020 -0.040  0.058 -0.087
month_end_london_after  -0.027 -0.028  0.058 -0.036  0.015  0.038  0.059  0.035 -0.011 -0.056  0.049  0.005  0.026 -0.035 -0.053
overnight_us            -0.038 -0.007 -0.008 -0.019 -0.030  0.022 -0.015  0.023  0.027 -0.002 -0.025 -0.007  0.031  0.009  0.028
europe_open_us          -0.049 -0.045 -0.039 -0.020 -0.026 -0.049 -0.022 -0.023 -0.001 -0.019 -0.009 -0.016 -0.007 -0.000  0.012
turn_of_month            0.073 -0.060 -0.057 -0.078 -0.122  0.047 -0.111 -0.018  0.114 -0.002 -0.019  0.063 -0.041 -0.002  0.105
```

Por mercado:

```
symbol                   AUDUSD  AUXAUD  EURUSD  FRXEUR  GBPUSD  GRXEUR  HKXHKD  JPXJPY  NSXUSD  NZDUSD  SPXUSD  UDXUSD  UKXGBP  USDCAD  USDCHF  USDJPY
fix_tokyo_before         -0.006     NaN  -0.006     NaN  -0.004     NaN     NaN     NaN     NaN  -0.008     NaN   0.002     NaN  -0.005  -0.011   0.000
fix_tokyo_after          -0.009     NaN  -0.005     NaN  -0.006     NaN     NaN     NaN     NaN  -0.013     NaN  -0.043     NaN  -0.008  -0.015   0.006
fix_ecb_before           -0.014     NaN   0.001     NaN  -0.008     NaN     NaN     NaN     NaN  -0.018     NaN  -0.050     NaN  -0.008  -0.010  -0.006
fix_ecb_after            -0.017     NaN  -0.007     NaN  -0.013     NaN     NaN     NaN     NaN  -0.020     NaN  -0.050     NaN  -0.022  -0.011  -0.012
fix_london_before        -0.018     NaN  -0.006     NaN  -0.010     NaN     NaN     NaN     NaN  -0.021     NaN  -0.049     NaN  -0.018  -0.015  -0.015
fix_london_after         -0.021     NaN  -0.008     NaN  -0.008     NaN     NaN     NaN     NaN  -0.021     NaN  -0.055     NaN  -0.011  -0.015  -0.015
month_end_london_before   0.002     NaN  -0.040     NaN  -0.017     NaN     NaN     NaN     NaN  -0.033     NaN  -0.080     NaN  -0.021  -0.025  -0.019
month_end_london_after    0.011     NaN   0.014     NaN   0.013     NaN     NaN     NaN     NaN   0.003     NaN  -0.020     NaN   0.006   0.010  -0.004
overnight_us                NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  -0.001     NaN  -0.002     NaN     NaN     NaN     NaN     NaN
europe_open_us              NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN  -0.023     NaN  -0.019     NaN     NaN     NaN     NaN     NaN
turn_of_month               NaN  -0.025     NaN  -0.009     NaN   0.004   -0.01  -0.042   0.010     NaN   0.004     NaN   -0.02     NaN     NaN     NaN
```
