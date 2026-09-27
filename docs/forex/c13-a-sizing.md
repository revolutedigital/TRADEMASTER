# C13, frente A: regras de dinheiro (exploratório, não é teste pré-registrado)

**Corrigido em 2026-09-27.** A primeira versão dimensionava cada operação sobre os US$ 1.000 iniciais mas media os tetos (stops abertos, por moeda) sobre o saldo, que crescia: com a conta maior cabiam cada vez mais operações, e o lucro por ano saiu muito inflado (US$ 2.000-2.800 por ano sobre US$ 1.000, quando o primeiro ano rendia ~20%). Agora cada simulação usa uma base só: o saldo do momento (`compostos` = True, o que o robô faz) ou os US$ 1.000 para sempre (`compostos` = False, lucro retirado e perda reposta).

Entradas e saídas congeladas (as escolhas do walk-forward, stop 1 ATR, alvo 3R, fim do dia FX); só muda quanto cada operação arrisca e quando uma operação é recusada. Conta de US$ 1.000, volume mínimo real de cada mercado na demo. `ao_ano`: retorno por ano (taxa composta quando reinveste, simples quando não); `primeiro_ano`: o que os US$ 1.000 rendem nos 12 primeiros meses do período; `pior_queda`: do saldo realizado; `pior_queda_todos_stops`: se todo stop aberto batesse junto. `teto_moeda`: a soma máxima de stops abertos numa mesma moeda e direção (inf = sem teto). `freio_dia`: sem novas entradas no resto do dia FX depois de perder essa fração no dia (inf = sem freio).

**Isto não aprova nada.** As regras são escolhidas só com 2016-2022 e lidas em 2023-2026.

Referência: 1% em tudo (a de hoje), stops abertos até 5%, sem teto por moeda, sem freio.

## reinvestindo (tudo sobre o saldo do momento — o que o robô faz)

### Uma regra de cada vez, a partir da referência

```
      regra                                valor   periodo  ao_ano  ao_ano_estresse  primeiro_ano  primeiro_ano_estresse  pior_queda  pior_queda_todos_stops  operacoes
    esquema               1% em tudo (a de hoje) 2016-2022  0.7551           0.4886         0.336                  0.287      0.2091                  0.2469       6801
    esquema                         1,5% em tudo 2016-2022  0.7247           0.4655       0.04494               -0.03315      0.2895                  0.3202       4673
    esquema                           2% em tudo 2016-2022  0.6727           0.4159       -0.2914                -0.3349      0.5131                   0.526       3467
    esquema        1% / 1,5% / 2% pela confiança 2016-2022  0.7193           0.4225       0.07152              -0.002191      0.2881                  0.3064       6252
    esquema          1% / 2% / 3% pela confiança 2016-2022  0.6886           0.4621       0.05004               -0.04676      0.3509                   0.377       5972
    esquema       0,5% abaixo de 0,05 + 1% acima 2016-2022  0.9489           0.5308        0.3862                 0.3324      0.1873                  0.2262      15955
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2016-2022   1.032           0.4089        0.4104                 0.1493      0.2341                  0.2681      15526
    esquema               1% em tudo (a de hoje) 2023-2026   1.385            1.022         0.453                 0.2749      0.2738                  0.3021       3622
    esquema                         1,5% em tudo 2023-2026   1.544            0.926        0.3281                0.01841      0.2998                  0.3347       2461
    esquema                           2% em tudo 2023-2026    1.52            1.062        0.1151               -0.05367      0.2896                  0.3243       1782
    esquema        1% / 1,5% / 2% pela confiança 2023-2026   1.535            1.112        0.3912                 0.2405      0.2749                   0.307       3326
    esquema          1% / 2% / 3% pela confiança 2023-2026   1.594             1.16         0.407                 0.2175       0.299                  0.3317       3190
    esquema       0,5% abaixo de 0,05 + 1% acima 2023-2026   1.825           0.9355         0.131                  0.248      0.2338                  0.2717       8763
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2023-2026   1.881           0.9513       0.08986               -0.05136      0.2367                  0.2747       8462
  freio_dia                                  inf 2016-2022  0.7551           0.4886         0.336                  0.287      0.2091                  0.2469       6801
  freio_dia                                 0.05 2016-2022  0.7501           0.5055         0.336                 0.2876       0.211                  0.2483       6793
  freio_dia                                 0.03 2016-2022  0.7275           0.4621        0.3235                 0.2305      0.2336                  0.2655       6685
  freio_dia                                 0.02 2016-2022  0.6433           0.4333        0.1232                0.08001      0.3193                  0.3487       6660
  freio_dia                                  inf 2023-2026   1.385            1.022         0.453                 0.2749      0.2738                  0.3021       3622
  freio_dia                                 0.05 2023-2026   1.366             1.03        0.4541                 0.2749      0.2737                  0.3028       3608
  freio_dia                                 0.03 2023-2026   1.354            1.001         0.453                 0.2957      0.2887                  0.3171       3571
  freio_dia                                 0.02 2023-2026   1.339           0.9502        0.3829                  0.292      0.3162                   0.334       3551
teto_aberto                                 0.05 2016-2022  0.7551           0.4886         0.336                  0.287      0.2091                  0.2469       6801
teto_aberto                                0.075 2016-2022    1.07           0.6189        0.4648                 0.1616      0.3029                  0.3508       8855
teto_aberto                                  0.1 2016-2022   1.367           0.7298         0.438                 0.1579      0.3799                  0.4412      10829
teto_aberto                                 0.05 2023-2026   1.385            1.022         0.453                 0.2749      0.2738                  0.3021       3622
teto_aberto                                0.075 2023-2026   2.173            1.398        0.5893                 0.3428      0.3219                  0.3636       4760
teto_aberto                                  0.1 2023-2026   3.228                2        0.7354                 0.3404      0.4034                  0.4567       5679
 teto_moeda                                  inf 2016-2022  0.7551           0.4886         0.336                  0.287      0.2091                  0.2469       6801
 teto_moeda                                 0.03 2016-2022  0.6611           0.4347        0.3574                 0.3057      0.2059                   0.245       6478
 teto_moeda                                 0.02 2016-2022  0.5549           0.3529        0.3497                 0.2874      0.1604                  0.1831       5792
 teto_moeda                                  inf 2023-2026   1.385            1.022         0.453                 0.2749      0.2738                  0.3021       3622
 teto_moeda                                 0.03 2023-2026   1.368            0.915        0.3353                  0.241      0.2379                  0.2738       3460
 teto_moeda                                 0.02 2023-2026   1.179           0.8272        0.2519                 0.2045       0.234                  0.2688       3132
```

### As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026

```
   posição                              esquema  teto_aberto  teto_moeda  freio_dia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_1º_ano  23-26_queda
        1ª       0,5% abaixo de 0,05 + 1% acima         0.05         inf        inf     0.9489          0.5308       0.2262      1.825          0.9355         0.131       0.2717
        2ª 0,5% / 1% / 1,5% / 2% pela confiança         0.05        0.03        inf     0.9378          0.5199       0.2416      1.693           1.087        0.0505       0.2903
        3ª               1% em tudo (a de hoje)         0.05         inf       0.05     0.7501          0.5055       0.2483      1.366            1.03        0.4541       0.3028
        4ª       0,5% abaixo de 0,05 + 1% acima         0.05        0.03       0.05     0.8882          0.4988       0.2432      1.703          0.9163       0.01622       0.2446
        5ª       0,5% abaixo de 0,05 + 1% acima         0.05        0.03        inf     0.9022          0.4988       0.2432      1.708           1.093       0.01622       0.2677
        6ª       0,5% abaixo de 0,05 + 1% acima         0.05         inf       0.02     0.9248          0.4951       0.2368      1.827          0.9934        0.3876       0.2414
        7ª               1% em tudo (a de hoje)         0.05         inf        inf     0.7551          0.4886       0.2469      1.385           1.022         0.453       0.3021
        8ª 0,5% / 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05      0.956            0.48       0.2365      1.688          0.9314        0.0505       0.2903
        9ª       0,5% abaixo de 0,05 + 1% acima         0.05        0.03       0.02     0.8625          0.4779       0.2124       1.67           1.066     -0.003343       0.2458
       10ª        1% / 1,5% / 2% pela confiança        0.075        0.03        inf      0.816          0.4698       0.2429      1.736           1.237        0.2624       0.3715
       11ª       0,5% abaixo de 0,05 + 1% acima         0.05         inf       0.03     0.9218          0.4684       0.2383      1.828          0.9361        0.4231       0.2398
       12ª        1% / 1,5% / 2% pela confiança         0.05         inf       0.03     0.6981          0.4545        0.249      1.465          0.9643         0.351       0.3248
referência               1% em tudo (a de hoje)         0.05         inf        inf     0.7551          0.4886       0.2469      1.385           1.022         0.453       0.3021
```

## sem reinvestir (tudo sobre os US$ 1.000 para sempre)

### Uma regra de cada vez, a partir da referência

```
      regra                                valor   periodo  ao_ano  ao_ano_estresse  primeiro_ano  primeiro_ano_estresse  pior_queda  pior_queda_todos_stops  operacoes
    esquema               1% em tudo (a de hoje) 2016-2022  0.5228           0.3703         0.332                 0.2308      0.1479                  0.1831       6771
    esquema                         1,5% em tudo 2016-2022  0.6016           0.4336       0.04585                 -0.071      0.2886                  0.3148       5499
    esquema                           2% em tudo 2016-2022  0.5626           0.3906      -0.06523                 -0.184      0.3635                  0.3942       4191
    esquema        1% / 1,5% / 2% pela confiança 2016-2022   0.532           0.3778        0.2502                 0.1463      0.1773                  0.2083       6289
    esquema          1% / 2% / 3% pela confiança 2016-2022  0.5961           0.4418        0.2055                 0.1021      0.2172                  0.2393       5936
    esquema       0,5% abaixo de 0,05 + 1% acima 2016-2022  0.5985           0.3899        0.5244                 0.3831      0.1821                   0.203      11113
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2016-2022  0.6227           0.4147        0.4325                 0.2919      0.1779                  0.2102      10529
    esquema               1% em tudo (a de hoje) 2023-2026   0.841           0.6556        0.3706                 0.2342      0.1414                  0.1836       3770
    esquema                         1,5% em tudo 2023-2026  0.8623           0.6623        0.1786                0.02258      0.2423                  0.2709       2838
    esquema                           2% em tudo 2023-2026  0.9923           0.7904        0.2121                0.05256      0.2368                  0.2833       2058
    esquema        1% / 1,5% / 2% pela confiança 2023-2026  0.8683           0.6811        0.3245                 0.1826      0.1906                   0.229       3442
    esquema          1% / 2% / 3% pela confiança 2023-2026   0.955           0.7664        0.3901                 0.2469      0.1712                  0.2145       3241
    esquema       0,5% abaixo de 0,05 + 1% acima 2023-2026   0.964            0.683        0.1901               -0.01198      0.2188                  0.2439       7092
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2023-2026   0.996           0.7147         0.232                0.02672      0.2297                   0.263       6771
  freio_dia                                  inf 2016-2022  0.5228           0.3703         0.332                 0.2308      0.1479                  0.1831       6771
  freio_dia                                 0.05 2016-2022  0.5239           0.3716        0.3271                 0.2266      0.1484                  0.1837       6764
  freio_dia                                 0.03 2016-2022  0.5335           0.3842        0.3732                 0.2762      0.1437                  0.1778       6732
  freio_dia                                 0.02 2016-2022  0.5231           0.3637        0.3877                  0.291      0.1445                  0.1783       6695
  freio_dia                                  inf 2023-2026   0.841           0.6556        0.3706                 0.2342      0.1414                  0.1836       3770
  freio_dia                                 0.05 2023-2026  0.8323           0.6423        0.3706                 0.2342      0.1414                  0.1836       3762
  freio_dia                                 0.03 2023-2026   0.832           0.6532        0.3783                 0.2435      0.1387                  0.1802       3753
  freio_dia                                 0.02 2023-2026  0.8298           0.6365         0.373                 0.2396      0.1322                  0.1737       3732
teto_aberto                                 0.05 2016-2022  0.5228           0.3703         0.332                 0.2308      0.1479                  0.1831       6771
teto_aberto                                0.075 2016-2022  0.7173           0.5174        0.4619                 0.3268      0.2049                  0.2625       8818
teto_aberto                                  0.1 2016-2022  0.8226           0.5871        0.5225                 0.3621      0.2561                  0.3369      10330
teto_aberto                                 0.05 2023-2026   0.841           0.6556        0.3706                 0.2342      0.1414                  0.1836       3770
teto_aberto                                0.075 2023-2026   1.155            0.909        0.5558                 0.3775       0.167                  0.2303       5017
teto_aberto                                  0.1 2023-2026    1.42            1.128        0.5822                 0.3739      0.1749                  0.2585       5969
 teto_moeda                                  inf 2016-2022  0.5228           0.3703         0.332                 0.2308      0.1479                  0.1831       6771
 teto_moeda                                 0.03 2016-2022  0.5005           0.3579        0.2789                 0.1887      0.1504                  0.1906       6341
 teto_moeda                                 0.02 2016-2022  0.4352           0.3093         0.328                 0.2486      0.1525                  0.1963       5564
 teto_moeda                                  inf 2023-2026   0.841           0.6556        0.3706                 0.2342      0.1414                  0.1836       3770
 teto_moeda                                 0.03 2023-2026   0.833           0.6571        0.3543                 0.2243      0.1379                  0.1757       3587
 teto_moeda                                 0.02 2023-2026  0.6951           0.5401        0.2632                 0.1474      0.1645                   0.192       3212
```

### As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026

```
   posição                     esquema  teto_aberto  teto_moeda  freio_dia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_1º_ano  23-26_queda
        1ª 1% / 2% / 3% pela confiança        0.075         inf       0.03     0.7992          0.5658       0.2414      1.166          0.9044         0.514       0.3515
        2ª 1% / 2% / 3% pela confiança         0.05         inf       0.03     0.6153           0.461       0.2357     0.9386          0.7595        0.3973       0.2115
        3ª      1% em tudo (a de hoje)        0.075        0.03       0.03     0.6216           0.457       0.2387     0.9718           0.759        0.3434       0.2462
        4ª                1,5% em tudo         0.05         inf       0.02     0.6176          0.4531       0.2425     0.8704          0.6869        0.1882        0.272
        5ª 1% / 2% / 3% pela confiança         0.05         inf       0.02     0.6154          0.4528       0.2489      0.945          0.7509         0.393        0.208
        6ª      1% em tudo (a de hoje)        0.075        0.03       0.05     0.6155          0.4445         0.24     0.9703          0.7554        0.3601       0.2402
        7ª 1% / 2% / 3% pela confiança         0.05         inf       0.05     0.5978          0.4439       0.2457     0.9284          0.7488        0.3901       0.2145
        8ª      1% em tudo (a de hoje)        0.075        0.03       0.02     0.6263          0.4428       0.2384     0.9679          0.7542         0.352       0.2429
        9ª 1% / 2% / 3% pela confiança         0.05         inf        inf     0.5961          0.4418       0.2393      0.955          0.7664        0.3901       0.2145
       10ª      1% em tudo (a de hoje)          0.1        0.03       0.03     0.6268          0.4396       0.2499      1.002          0.7682        0.2563       0.2828
       11ª      1% em tudo (a de hoje)        0.075        0.03        inf     0.6107          0.4383       0.2447     0.9931          0.7775        0.3601       0.2402
       12ª      1% em tudo (a de hoje)          0.1        0.03       0.02     0.6388          0.4377       0.2495      1.004          0.7903        0.2611       0.2828
referência      1% em tudo (a de hoje)         0.05         inf        inf     0.5228          0.3703       0.1831      0.841          0.6556        0.3706       0.1836
```

