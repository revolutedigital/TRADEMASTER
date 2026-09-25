# C10: descoberta (2012-01 a 2026-08, 37 mercados, 3,798 dias de pregão)

Commit do código: 7d2fee604934512a1c7c3a6b84a839265f67c43d. Configurações testadas: 4,464. **Aprovadas na etapa A (p máx-t ≤ 0,05 e o resto): 0.** Unidade: R (a distância do stop inicial), líquido do custo; com 0,5% de risco por operação numa conta de US$ 1.000, 1 R = US$ 5.

## As 25 melhores por t

```
                      set                                                             management  trades  trades_per_day  win_rate     mean_r  mean_r_stress       t  p_maxt  market_share  avg_win_r  avg_loss_r  mean_minutes
bollinger M15 (width=2.0)                                stop 1 ATR · alvo 3R · até o fim do dia  150268           39.57    0.5002   0.003059       -0.01962  0.9979       1        0.6486     0.4615     -0.4558          1089
bollinger M15 (width=2.0)                                stop 1 ATR · alvo 2R · até o fim do dia  151063           39.77    0.5002   0.002592       -0.02009  0.8541       1        0.6216     0.4607      -0.456          1082
bollinger M15 (width=2.0)              stop 1 ATR · metade em 1R, resto em 3R · até o fim do dia  150862           39.72    0.5031    0.00228        -0.0204  0.7649       1        0.6216     0.4545     -0.4555          1084
bollinger M15 (width=2.0)              stop 1 ATR · metade em 1R, resto em 6R · até o fim do dia  150630           39.66     0.503   0.002293       -0.02038    0.76       1        0.6216     0.4541      -0.455          1086
bollinger M15 (width=2.0)    stop 1 ATR · metade em 1R, resto com trailing 1R · até o fim do dia  151446           39.88    0.5031   0.002036       -0.02064   0.683       1        0.6216     0.4537     -0.4553          1079
bollinger M15 (width=2.0)  stop 1 ATR · metade em 1R, resto com trailing 0.5R · até o fim do dia  154816           40.76    0.5025   0.001412       -0.02127  0.4834       1        0.5946     0.4504     -0.4521          1048
bollinger M15 (width=2.0)                                stop 1 ATR · alvo 1R · até o fim do dia  164357           43.27     0.502  0.0003024        -0.0224  0.1081       1        0.6216     0.4434     -0.4464         985.6
bollinger M15 (width=2.0)            stop 1 ATR · metade em 0.5R, resto em 6R · até o fim do dia  161856           42.62    0.5483  0.0002222       -0.02247 0.08526       1        0.6216     0.3767     -0.4568          1001
bollinger M15 (width=2.0)            stop 1 ATR · metade em 0.5R, resto em 3R · até o fim do dia  162099           42.68    0.5484  8.749e-05       -0.02261 0.03393       1        0.6216     0.3767     -0.4572         999.2
bollinger M15 (width=2.0)  stop 1 ATR · metade em 0.5R, resto com trailing 1R · até o fim do dia  162706           42.84    0.5485  2.785e-05       -0.02266 0.01079       1        0.5946     0.3761     -0.4569         994.4
bollinger M15 (width=2.0) stop 1 ATR · metade em 0.5R, resto com trailing 0.5R · até o fim do...  173051           45.56    0.5449 -0.0009392       -0.02364 -0.3842       1        0.5946       0.37      -0.445         919.4
bollinger M15 (width=2.5)                                stop 1 ATR · alvo 3R · até o fim do dia  140540              37    0.4957  -0.002405       -0.02506 -0.8181       1        0.5946     0.4405     -0.4378         970.5
bollinger M15 (width=2.5)                                stop 1 ATR · alvo 2R · até o fim do dia  140948           37.11    0.4957  -0.003143        -0.0258  -1.078       1        0.5676     0.4396     -0.4383         966.5
bollinger M15 (width=2.5)              stop 1 ATR · metade em 1R, resto em 6R · até o fim do dia  140661           37.04    0.4981  -0.003256       -0.02591  -1.127       1        0.5676     0.4343     -0.4376         968.9
bollinger M15 (width=2.5)              stop 1 ATR · metade em 1R, resto em 3R · até o fim do dia  140782           37.07    0.4981  -0.003399       -0.02605  -1.184       1        0.5676     0.4344     -0.4379         967.8
bollinger M15 (width=2.5)    stop 1 ATR · metade em 1R, resto com trailing 1R · até o fim do dia  141006           37.13    0.4981  -0.003565       -0.02622  -1.243       1        0.5676     0.4341      -0.438         964.8
bollinger M15 (width=2.0)                              stop 0.5 ATR · alvo 3R · até o fim do dia  198229           52.19    0.4605  -0.005555       -0.05082  -1.272       1        0.5135     0.8784     -0.7602         804.3
bollinger M15 (width=2.5)  stop 1 ATR · metade em 1R, resto com trailing 0.5R · até o fim do dia  142396           37.49    0.4981  -0.004103       -0.02676  -1.444       1        0.5676     0.4325     -0.4374         946.8
bollinger M15 (width=2.5)            stop 1 ATR · metade em 0.5R, resto em 6R · até o fim do dia  146001           38.44    0.5397  -0.004617       -0.02728  -1.772       1        0.5405     0.3695     -0.4433         914.8
bollinger M15 (width=2.5)                                stop 1 ATR · alvo 1R · até o fim do dia  147398           38.81    0.4979  -0.005012       -0.02768  -1.792       1        0.5405     0.4289     -0.4352           904
bollinger M15 (width=2.5)            stop 1 ATR · metade em 0.5R, resto em 3R · até o fim do dia  146118           38.47    0.5398  -0.004709       -0.02738   -1.82       1        0.5135     0.3694     -0.4435         913.7
bollinger M15 (width=2.5)  stop 1 ATR · metade em 0.5R, resto com trailing 1R · até o fim do dia  146338           38.53    0.5399  -0.004782       -0.02745  -1.843       1        0.5135     0.3691     -0.4435           911
bollinger M15 (width=2.5) stop 1 ATR · metade em 0.5R, resto com trailing 0.5R · até o fim do...  150787            39.7    0.5389  -0.005327         -0.028  -2.106       1        0.5405      0.366     -0.4392         860.9
     gap M5 (gap_atr=0.5)                                stop 1 ATR · alvo 2R · até o fim do dia   25354           6.676    0.4796   -0.01859       -0.04141  -2.215       1        0.2162     0.4551     -0.4552         789.4
bollinger M15 (width=2.0)            stop 0.5 ATR · metade em 1R, resto em 6R · até o fim do dia  206189           54.29    0.5047  -0.008389       -0.05364  -2.223       1        0.5135     0.7331     -0.7638         767.6
```

## A melhor de cada família e parâmetro

```
                                  set                                   management  trades  trades_per_day  win_rate    mean_r  mean_r_stress       t  p_maxt  market_share  avg_win_r  avg_loss_r  mean_minutes
            bollinger M15 (width=2.0)      stop 1 ATR · alvo 3R · até o fim do dia  150268           39.57    0.5002  0.003059       -0.01962  0.9979       1        0.6486     0.4615     -0.4558          1089
            bollinger M15 (width=2.5)      stop 1 ATR · alvo 3R · até o fim do dia  140540              37    0.4957 -0.002405       -0.02506 -0.8181       1        0.5946     0.4405     -0.4378         970.5
                 gap M5 (gap_atr=0.5)      stop 1 ATR · alvo 2R · até o fim do dia   25354           6.676    0.4796  -0.01859       -0.04141  -2.215       1        0.2162     0.4551     -0.4552         789.4
           rsi2 M15 (low=10, high=90)      stop 1 ATR · alvo 3R · até o fim do dia  145580           38.33    0.4848 -0.009466        -0.0321  -2.896       1        0.2162     0.4332      -0.426          1046
                soup M15 (lookback=1)      stop 1 ATR · alvo 3R · até o fim do dia  107871            28.4    0.4872  -0.01108       -0.03378  -3.255       1        0.1081      0.404     -0.4054           892
       burst M5 (size=4.0, mode=fade)      stop 1 ATR · alvo 3R · até o fim do dia   31221            8.22    0.4869  -0.01701       -0.04005   -3.36       1        0.4324     0.3827     -0.3963         595.9
            bollinger M30 (width=2.0)      stop 1 ATR · alvo 3R · até o fim do dia  145309           38.26    0.4894  -0.01075       -0.03342   -3.49       1        0.1892     0.4287      -0.432         920.5
               soup M15 (lookback=20)      stop 1 ATR · alvo 3R · até o fim do dia    8078           2.127    0.4673  -0.02748       -0.05085  -3.783       1        0.2162     0.3616     -0.3688         610.2
            rsi2 M30 (low=5, high=95)      stop 1 ATR · alvo 2R · até o fim do dia   96972           25.53    0.4851   -0.0128       -0.03526  -3.876       1        0.1892     0.3744     -0.3776           819
                gap M5 (gap_atr=0.25)      stop 1 ATR · alvo 2R · até o fim do dia   65482           17.24    0.4788  -0.01979       -0.04246  -3.993       1        0.1081     0.4248     -0.4282         792.1
           rsi2 M30 (low=10, high=90)      stop 1 ATR · alvo 3R · até o fim do dia  131086           34.51    0.4832  -0.01303       -0.03558  -4.064       1        0.1892     0.4041     -0.4031         927.9
     burst M5 (size=4.0, mode=follow)           stop 1 ATR · alvo 3R · máx 240 min   32364           8.521    0.4668  -0.01851       -0.04159  -4.173       1        0.1622     0.3053      -0.302         233.9
            rsi2 M15 (low=5, high=95)      stop 1 ATR · alvo 3R · até o fim do dia  129431           34.08    0.4813  -0.01374       -0.03629  -4.358       1        0.2162     0.3976     -0.3954         908.3
            bollinger M30 (width=2.5)      stop 1 ATR · alvo 3R · até o fim do dia  118812           31.28     0.487  -0.01423       -0.03681  -4.718       1        0.1081     0.3994     -0.4068         795.4
                             asian M5      stop 1 ATR · alvo 3R · até o fim do dia  109081           28.72    0.4796  -0.01852        -0.0412   -5.45       1        0.1622     0.3987     -0.4031         786.4
                   donchian H1 (n=55)      stop 1 ATR · alvo 3R · até o fim do dia   46010           12.11    0.4609  -0.03124       -0.05385  -5.867       1        0.1622     0.3822     -0.3848         752.9
                          nr M5 (n=7) stop estrutural · alvo 3R · até o fim do dia   19961           5.256    0.4528  -0.05179       -0.08934  -5.876       1        0.2432     0.6571     -0.6384         843.4
            orb M5 (range_minutes=60)      stop 1 ATR · alvo 3R · até o fim do dia  124651           32.82    0.4773  -0.01735        -0.0399  -6.404       1       0.08108     0.3681     -0.3693           704
                  williams M5 (k=0.7)      stop 1 ATR · alvo 3R · até o fim do dia   69917           18.41    0.4589  -0.02596       -0.04891  -6.467       1        0.1892     0.3565     -0.3504         619.5
            orb M5 (range_minutes=30)      stop 1 ATR · alvo 3R · até o fim do dia  134296           35.36    0.4793  -0.01699       -0.03957  -6.517       1       0.08108     0.3816      -0.384         740.6
orb M5 (range_minutes=30, trend=True)      stop 1 ATR · alvo 3R · até o fim do dia   84983           22.38     0.477  -0.02285       -0.04537  -6.517       1             0     0.3715     -0.3825         722.8
                               pdh M5      stop 1 ATR · alvo 3R · até o fim do dia  101069           26.61    0.4706  -0.02477       -0.04737  -6.979       1       0.08108     0.3739     -0.3793         678.2
            orb M5 (range_minutes=15)      stop 1 ATR · alvo 3R · até o fim do dia  137088           36.09     0.478  -0.01951       -0.04211  -7.416       1       0.05405     0.3905     -0.3949         763.1
                          nr M5 (n=4) stop estrutural · alvo 3R · até o fim do dia   33444           8.806    0.4558  -0.05141       -0.08559  -7.453       1        0.1622     0.5993     -0.5965           848
                   donchian H1 (n=20)      stop 1 ATR · alvo 2R · até o fim do dia   70863           18.66    0.4614   -0.0327       -0.05513  -7.805       1       0.02703     0.3712     -0.3787         744.8
                  williams M5 (k=0.5)    stop 0.5 ATR · alvo 3R · até o fim do dia  100142           26.37    0.4429  -0.04921       -0.09486  -8.006       1        0.1622      0.754     -0.6877           602
           orb M5 (first_candle=True)      stop 1 ATR · alvo 1R · até o fim do dia  136089           35.83    0.4812  -0.02085        -0.0434  -8.012       1       0.02703     0.4013     -0.4125         765.2
       burst M5 (size=2.5, mode=fade)      stop 1 ATR · alvo 3R · até o fim do dia  101606           26.75    0.4827  -0.02101       -0.04339  -9.108       1       0.05405     0.3603     -0.3768           627
     burst M5 (size=2.5, mode=follow)      stop 1 ATR · alvo 3R · até o fim do dia  101336           26.68    0.4678  -0.02275       -0.04514  -9.132       1       0.05405     0.3704     -0.3684         630.3
                  donchian M15 (n=55)      stop 1 ATR · alvo 3R · até o fim do dia   95291           25.09    0.4614  -0.03643       -0.05894  -9.575       1       0.02703     0.3903      -0.402         812.1
                  williams M5 (k=0.3)      stop 1 ATR · alvo 3R · até o fim do dia  129559           34.11     0.467  -0.03234       -0.05504  -10.06       1       0.02703      0.415     -0.4243           895
                  donchian M15 (n=20)      stop 1 ATR · alvo 3R · até o fim do dia  130096           34.25    0.4616  -0.04041       -0.06298  -12.05       1             0     0.4109     -0.4273         932.2
             momentum M5 (rule=first)      stop 1 ATR · alvo 3R · até o fim do dia  133915           35.26    0.4566  -0.02258       -0.04519  -13.19       1             0     0.2019     -0.2112         360.5
              momentum M5 (rule=both)      stop 1 ATR · alvo 3R · até o fim do dia   66065           17.39    0.4344  -0.03197       -0.05436  -14.09       1             0     0.2018     -0.2115         360.4
```

## Levadas ao dado intocado (2003 a 2011)

```
Empty DataFrame
Columns: [config, set, management, trades, trades_per_day, win_rate, mean_r, mean_r_stress, t, p_maxt, market_share, avg_win_r, avg_loss_r, mean_minutes]
Index: []
```

Média por ano (R por operação):

```
{}
```
