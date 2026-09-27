# C13, frente A: regras de dinheiro, rodada 2 (exploratório, não é teste pré-registrado)

**Corrigido em 2026-09-27.** A primeira versão dimensionava cada operação sobre os US$ 1.000 iniciais mas media os tetos (stops abertos, por moeda) sobre o saldo, que crescia: com a conta maior cabiam cada vez mais operações, e o lucro por ano saiu muito inflado (US$ 2.000-2.800 por ano sobre US$ 1.000, quando o primeiro ano rendia ~20%). Agora cada simulação usa uma base só: o saldo do momento (`compostos` = True, o que o robô faz) ou os US$ 1.000 para sempre (`compostos` = False, lucro retirado e perda reposta).

Duas regras novas: `prioridade` — operações que entram no mesmo minuto recebem a vaga dos tetos em ordem da previsão do modelo, maior primeiro; `tolerancia` — quando o volume mínimo arrisca mais que o orçamento, entra com o mínimo se ele arriscar até essa vez o orçamento (1 = pula).

Entradas e saídas congeladas (as escolhas do walk-forward, stop 1 ATR, alvo 3R, fim do dia FX); só muda quanto cada operação arrisca e quando uma operação é recusada. Conta de US$ 1.000, volume mínimo real de cada mercado na demo. `ao_ano`: retorno por ano (taxa composta quando reinveste, simples quando não); `primeiro_ano`: o que os US$ 1.000 rendem nos 12 primeiros meses do período; `pior_queda`: do saldo realizado; `pior_queda_todos_stops`: se todo stop aberto batesse junto. `teto_moeda`: a soma máxima de stops abertos numa mesma moeda e direção (inf = sem teto). `freio_dia`: sem novas entradas no resto do dia FX depois de perder essa fração no dia (inf = sem freio).

**Isto não aprova nada.** As regras são escolhidas só com 2016-2022 e lidas em 2023-2026.

Referência: as regras que rodam na demo (1% / 2% / 3% pela confiança, stops abertos até 7,5%, 3% por moeda, freio a -5% do dia, prioridade).

## reinvestindo (tudo sobre o saldo do momento — o que o robô faz)

### Uma regra de cada vez, a partir da referência

```
      regra                         valor   periodo  ao_ano  ao_ano_estresse  primeiro_ano  primeiro_ano_estresse  pior_queda  pior_queda_todos_stops  operacoes
    esquema   1% / 2% / 3% pela confiança 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
    esquema 1% / 1,5% / 2% pela confiança 2016-2022  0.7835           0.4492         0.252                0.06997      0.2642                  0.3141       7065
    esquema        1% em tudo (a de hoje) 2016-2022  0.8073           0.4127         0.381                 0.1826      0.2358                  0.2812       7718
    esquema   1% / 2% / 3% pela confiança 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
    esquema 1% / 1,5% / 2% pela confiança 2023-2026   1.807            1.303        0.2129                 0.1268      0.2371                  0.2896       3800
    esquema        1% em tudo (a de hoje) 2023-2026   1.757            1.236        0.2307                 0.1415      0.2428                   0.297       4262
  freio_dia                           inf 2016-2022  0.7142           0.4455       0.07388                0.06357      0.2697                  0.3026       6730
  freio_dia                          0.05 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
  freio_dia                           inf 2023-2026   1.897            1.378        0.1566                0.05051      0.3026                   0.352       3634
  freio_dia                          0.05 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
 prioridade                         False 2016-2022  0.6586           0.4024        0.1107                0.03323      0.2257                  0.2782       6917
 prioridade                          True 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
 prioridade                         False 2023-2026   1.762            1.259        0.2244                 0.1396      0.2272                  0.2817       3770
 prioridade                          True 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
teto_aberto                          0.05 2016-2022  0.6471           0.3448        0.4164                 0.1539      0.2287                  0.2614       5614
teto_aberto                         0.075 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
teto_aberto                           0.1 2016-2022  0.7414           0.4005        0.2899                0.04297      0.2934                   0.361       7404
teto_aberto                          0.05 2023-2026   1.438             1.13        0.1996                  0.128      0.2079                  0.2438       2966
teto_aberto                         0.075 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
teto_aberto                           0.1 2023-2026   2.133            1.389        0.2062               0.002428      0.2352                  0.2931       3996
 teto_moeda                           inf 2016-2022   1.254           0.6752         0.304                 -0.113      0.3128                  0.3604       7313
 teto_moeda                          0.03 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
 teto_moeda                          0.02 2016-2022  0.5439           0.3012        0.3689                 0.1934      0.2164                  0.2628       5776
 teto_moeda                           inf 2023-2026   2.903            1.799         0.387                 0.2214      0.3121                  0.3577       3834
 teto_moeda                          0.03 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
 teto_moeda                          0.02 2023-2026   1.195           0.7825        0.2318                0.08479      0.2102                  0.2629       3108
 tolerancia                             1 2016-2022  0.8084           0.4103        0.5347                0.06195      0.2273                  0.2734       6722
 tolerancia                           1.5 2016-2022  0.7299           0.4117        0.1803               0.008844      0.2403                  0.2751       6824
 tolerancia                             1 2023-2026   1.859            1.225        0.1602                0.05469      0.2486                  0.3006       3625
 tolerancia                           1.5 2023-2026   1.864            1.186        0.1298               -0.05933      0.2578                  0.3071       3647
```

### As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026

```
   posição                       esquema  teto_aberto  teto_moeda  freio_dia  prioridade  tolerancia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_1º_ano  23-26_queda
        1ª        1% em tudo (a de hoje)         0.05         inf       0.05       False           1     0.7501          0.5055       0.2483      1.366            1.03        0.4541       0.3028
        2ª        1% em tudo (a de hoje)         0.05         inf        inf       False           1     0.7551          0.4886       0.2469      1.385           1.022         0.453       0.3021
        3ª 1% / 1,5% / 2% pela confiança        0.075        0.03        inf       False           1      0.816          0.4698       0.2429      1.736           1.237        0.2624       0.3715
        4ª        1% em tudo (a de hoje)         0.05        0.03       0.05       False           1     0.6554          0.4347       0.2422      1.356          0.9482        0.3353       0.2515
        5ª        1% em tudo (a de hoje)         0.05        0.03        inf       False           1     0.6611          0.4347        0.245      1.368           0.915        0.3353       0.2738
        6ª 1% / 1,5% / 2% pela confiança         0.05        0.03        inf       False           1     0.6259           0.434       0.2303      1.451           1.044        0.3112       0.2787
        7ª 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05       False           1      0.626          0.4337       0.2304      1.383          0.9613        0.3112       0.2475
        8ª        1% em tudo (a de hoje)         0.05        0.03       0.05        True           1     0.6773           0.426        0.246        1.7           1.097        0.2877       0.2495
        9ª        1% em tudo (a de hoje)         0.05        0.03        inf        True           1     0.6803          0.4255       0.2451      1.681           1.097        0.2877       0.2653
       10ª   1% / 2% / 3% pela confiança         0.05        0.03        inf       False           1     0.6252          0.4094       0.2464      1.306          0.9407        0.2395       0.2867
       11ª 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05        True           1     0.5982          0.4034       0.2382      1.461           1.139        0.2069       0.2693
       12ª        1% em tudo (a de hoje)         0.05        0.03       0.05        True         1.5     0.6416          0.3907       0.2484      1.449           1.031        0.1753       0.2531
referência   1% / 2% / 3% pela confiança        0.075        0.03       0.05        True           1     0.8084          0.4103       0.2734      1.859           1.225        0.1602       0.3006
```

## sem reinvestir (tudo sobre os US$ 1.000 para sempre)

### Uma regra de cada vez, a partir da referência

```
      regra                         valor   periodo  ao_ano  ao_ano_estresse  primeiro_ano  primeiro_ano_estresse  pior_queda  pior_queda_todos_stops  operacoes
    esquema   1% / 2% / 3% pela confiança 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
    esquema 1% / 1,5% / 2% pela confiança 2016-2022  0.6173           0.4449        0.3088                 0.2001      0.1979                  0.2601       6949
    esquema        1% em tudo (a de hoje) 2016-2022  0.5937           0.4234        0.4822                 0.3826       0.175                  0.2332       7534
    esquema   1% / 2% / 3% pela confiança 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
    esquema 1% / 1,5% / 2% pela confiança 2023-2026   0.993            0.778        0.2796                 0.1231      0.2009                  0.2693       3926
    esquema        1% em tudo (a de hoje) 2023-2026  0.9714             0.76        0.3012                 0.1485      0.1885                  0.2479       4425
  freio_dia                           inf 2016-2022  0.5825           0.4088           0.3                 0.1839      0.2095                  0.2564       6548
  freio_dia                          0.05 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
  freio_dia                           inf 2023-2026   1.046             0.83        0.2237                0.06365      0.1756                  0.2433       3675
  freio_dia                          0.05 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
 prioridade                         False 2016-2022  0.5817           0.4076        0.2905                 0.1778      0.1826                  0.2296       6765
 prioridade                          True 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
 prioridade                         False 2023-2026   1.009           0.8014         0.382                 0.2228      0.1539                  0.2214       3828
 prioridade                          True 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
teto_aberto                          0.05 2016-2022  0.5273           0.3868        0.2948                 0.2146      0.1561                  0.1757       5473
teto_aberto                         0.075 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
teto_aberto                           0.1 2016-2022  0.6177           0.4346        0.3528                 0.2633      0.2333                  0.2989       7026
teto_aberto                          0.05 2023-2026  0.9143             0.74        0.2776                 0.1462      0.1514                  0.1944       2986
teto_aberto                         0.075 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
teto_aberto                           0.1 2023-2026   1.105           0.8618         0.278                 0.1078      0.2387                  0.3171       4035
 teto_moeda                           inf 2016-2022  0.8589           0.6536        0.3011                 0.1692      0.2747                  0.3047       7373
 teto_moeda                          0.03 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
 teto_moeda                          0.02 2016-2022  0.4504           0.3098        0.3342                  0.237      0.1873                  0.2346       5500
 teto_moeda                           inf 2023-2026   1.359            1.093        0.5359                 0.3438      0.2932                  0.3618       4033
 teto_moeda                          0.03 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
 teto_moeda                          0.02 2023-2026  0.7106           0.5336        0.2485                 0.1211      0.1803                  0.2505       3091
 tolerancia                             1 2016-2022  0.5904           0.4163        0.3549                 0.2367      0.2063                  0.2525       6542
 tolerancia                           1.5 2016-2022  0.5986            0.411         0.383                 0.2409      0.2411                  0.2891       7100
 tolerancia                             1 2023-2026    1.01           0.8076        0.2237                0.06365      0.1756                  0.2433       3670
 tolerancia                           1.5 2023-2026  0.9946             0.79        0.1446               -0.01475      0.2047                  0.2735       3804
```

### As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026

```
   posição                     esquema  teto_aberto  teto_moeda  freio_dia  prioridade  tolerancia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_1º_ano  23-26_queda
        1ª      1% em tudo (a de hoje)        0.075         inf       0.05        True           1     0.7577          0.5606       0.2494      1.201          0.9569        0.4909        0.234
        2ª 1% / 2% / 3% pela confiança         0.05         inf       0.05        True         1.5     0.6391           0.479       0.2355      1.102          0.9137        0.2543       0.2088
        3ª 1% / 2% / 3% pela confiança         0.05         inf       0.05        True           1     0.6209          0.4683       0.2445      1.071          0.8845        0.4204       0.2266
        4ª 1% / 2% / 3% pela confiança         0.05         inf        inf        True         1.5     0.6249          0.4642       0.2363      1.112           0.923        0.2543       0.2088
        5ª 1% / 2% / 3% pela confiança         0.05         inf        inf        True           1     0.6046          0.4515       0.2486      1.081          0.8938        0.4204       0.2266
        6ª      1% em tudo (a de hoje)        0.075        0.03       0.05       False           1     0.6155          0.4445         0.24     0.9703          0.7554        0.3601       0.2402
        7ª 1% / 2% / 3% pela confiança         0.05         inf       0.05       False           1     0.5978          0.4439       0.2457     0.9284          0.7488        0.3901       0.2145
        8ª 1% / 2% / 3% pela confiança         0.05         inf        inf       False           1     0.5961          0.4418       0.2393      0.955          0.7664        0.3901       0.2145
        9ª      1% em tudo (a de hoje)          0.1        0.03       0.05        True           1     0.6253          0.4401       0.2436      1.051          0.8127        0.2892       0.2622
       10ª 1% / 2% / 3% pela confiança         0.05         inf       0.05       False         1.5     0.6022          0.4384       0.2496     0.9514          0.7628        0.2373       0.1988
       11ª      1% em tudo (a de hoje)        0.075        0.03        inf       False           1     0.6107          0.4383       0.2447     0.9931          0.7775        0.3601       0.2402
       12ª 1% / 2% / 3% pela confiança         0.05         inf        inf       False         1.5     0.5996          0.4383        0.249     0.9688          0.7796        0.2373       0.2166
referência 1% / 2% / 3% pela confiança        0.075        0.03       0.05        True           1     0.5904          0.4163       0.2525       1.01          0.8076        0.2237       0.2433
```

