# C13, frente A: varredura exploratória de limiar, gestão e risco (não é teste pré-registrado)

**Corrigido em 2026-09-27.** A primeira versão dimensionava cada operação sobre os US$ 1.000 iniciais mas media o teto de stops abertos sobre o saldo, que crescia: com a conta maior cabiam cada vez mais operações, e o lucro por ano "sem juros compostos" saiu muito inflado. Agora o teto usa a mesma base do tamanho.

As candidatas que o walk-forward já decidiu em 2016-2026 (cada ano por um modelo que só viu operações encerradas antes dele), refeitas com 5 limiares, 40 gestões e 5 riscos, em dinheiro a partir de US$ 1,000 com o volume mínimo real de cada mercado na demo. A gestão congelada no limiar congelado reproduz a descoberta operação por operação (checado na execução). Operações com |R| > 15 num mesmo dia são vela ruim no dado e ficam de fora (contadas).

**Isto não aprova nada.** A melhor linha entre centenas, medida no mesmo dado que a escolhe, é otimista por construção. O teste honesto é a última tabela: escolhida só com 2016-2022, olhada em 2023-2026.

## 1. O limiar do modelo, com a gestão congelada (stop 1 ATR · alvo 3R)

```
 limiar               gestao   periodo  trades  por_dia  media_R  media_R_estresse  acerto      t  total_R  pior_queda_R  descartadas_dado_ruim
   0.02 stop 1 ATR · alvo 3R 2016-2022   95183     52.3  0.03989           0.02173  0.5247  4.982    3,797         372.3                      0
   0.05 stop 1 ATR · alvo 3R 2016-2022   24077    13.23  0.05941           0.04115  0.5309  3.308    1,430         362.4                      0
   0.08 stop 1 ATR · alvo 3R 2016-2022    8409     4.62  0.08387            0.0654  0.5426   2.11    705.2         348.4                      0
   0.12 stop 1 ATR · alvo 3R 2016-2022    3522    1.935  0.09962           0.08206  0.5329   1.39    350.8         244.7                      0
   0.16 stop 1 ATR · alvo 3R 2016-2022    1743   0.9577  0.09534           0.07855   0.529 0.9073    166.2         137.6                      0
   0.02 stop 1 ATR · alvo 3R 2016-2026  162728    58.66  0.04473           0.02532  0.5254  7.148    7,279           373                      0
   0.05 stop 1 ATR · alvo 3R 2016-2026   36735    13.24  0.08271           0.06327  0.5431  5.661    3,038         362.8                      0
   0.08 stop 1 ATR · alvo 3R 2016-2026   12808    4.617   0.1247            0.1052  0.5626  3.924    1,598         348.7                      0
   0.12 stop 1 ATR · alvo 3R 2016-2026    5318    1.917   0.1685            0.1496  0.5681   2.98    895.9         244.1                      0
   0.16 stop 1 ATR · alvo 3R 2016-2026    2595   0.9355    0.211            0.1934  0.5746  2.418    547.5         137.6                      0
   0.02 stop 1 ATR · alvo 3R 2023-2026   67545     70.8  0.05156           0.03038  0.5264  5.152    3,482         235.2                      0
   0.05 stop 1 ATR · alvo 3R 2023-2026   12658    13.27    0.127            0.1054  0.5664  5.203    1,608         109.4                      0
   0.08 stop 1 ATR · alvo 3R 2023-2026    4399    4.611   0.2029            0.1812  0.6008  4.157    892.4         82.59                      0
   0.12 stop 1 ATR · alvo 3R 2023-2026    1796    1.883   0.3035             0.282   0.637  3.912    545.1         60.75                      0
   0.16 stop 1 ATR · alvo 3R 2023-2026     852   0.8931   0.4476            0.4284  0.6678  3.783    381.3         41.26                      0
```

## 2. As 40 gestões no limiar congelado (0.05)

```
 limiar                                       gestao   periodo  trades  por_dia  media_R  media_R_estresse  acerto     t  total_R  pior_queda_R  descartadas_dado_ruim
   0.05                       stop 0.5 ATR · alvo 5R 2016-2026   36735    13.24   0.1396            0.1007  0.4895 6.091    5,128         519.6                      0
   0.05                       stop 0.5 ATR · alvo 3R 2016-2026   36735    13.24   0.1341           0.09524  0.4909 5.977    4,926         479.5                      0
   0.05                       stop 0.5 ATR · alvo 2R 2016-2026   36735    13.24   0.1206           0.08172  0.4973 5.697    4,430         469.7                      0
   0.05    stop 0.5 ATR · metade em 1R + resto em 3R 2016-2026   36735    13.24   0.1028           0.06398  0.5615 5.705    3,778         400.3                      0
   0.05  stop 0.5 ATR · empate em 1R + trailing 0,5R 2016-2026   36735    13.24   0.1017           0.06285  0.5615 5.819    3,737         289.9                      0
   0.05    stop 0.5 ATR · empate em 1R + trailing 1R 2016-2026   36735    13.24   0.1016           0.06278  0.5466 5.277    3,734         317.9                      0
   0.05 stop 0.5 ATR · empate em 1R + trailing 0,25R 2016-2026   36735    13.24   0.1002           0.06133  0.5615 6.074    3,681           357                      0
   0.05  stop 0.5 ATR · metade em 1R + trailing 0,5R 2016-2026   36735    13.24  0.09581           0.05694  0.5615 5.749    3,520         341.5                      0
   0.05    stop 0.5 ATR · metade em 1R + trailing 1R 2016-2026   36735    13.24  0.09578           0.05691  0.5615 5.543    3,518         324.8                      0
   0.05                       stop 0.5 ATR · alvo 1R 2016-2026   36735    13.24  0.08991           0.05104  0.5615 5.572    3,303           393                      0
   0.05                         stop 1 ATR · alvo 3R 2016-2026   36735    13.24  0.08271           0.06327  0.5431 5.661    3,038         362.8                      0
   0.05                         stop 1 ATR · alvo 5R 2016-2026   36704    13.23  0.08144             0.062   0.543   5.6    2,989         375.5                      0
   0.05                         stop 1 ATR · alvo 2R 2016-2026   36735    13.24  0.08109           0.06165  0.5433 5.468    2,979         376.4                      0
   0.05      stop 1 ATR · metade em 1R + resto em 3R 2016-2026   36735    13.24  0.07409           0.05465  0.5516  5.49    2,722           379                      0
   0.05   stop 1 ATR · empate em 1R + trailing 0,25R 2016-2026   36735    13.24  0.07185           0.05242  0.5516 5.416    2,639         319.5                      0
   0.05      stop 1 ATR · empate em 1R + trailing 1R 2016-2026   36734    13.24  0.07151           0.05207  0.5502 5.274    2,627         374.6                      0
   0.05      stop 1 ATR · metade em 1R + trailing 1R 2016-2026   36734    13.24  0.07074           0.05131  0.5516 5.302    2,599         380.5                      0
   0.05                         stop 1 ATR · alvo 1R 2016-2026   36735    13.24  0.06998           0.05055  0.5516 5.242    2,571         365.5                      0
   0.05    stop 1 ATR · metade em 1R + trailing 0,5R 2016-2026   36735    13.24  0.06984            0.0504  0.5516 5.246    2,565           351                      0
   0.05    stop 1 ATR · empate em 1R + trailing 0,5R 2016-2026   36735    13.24  0.06969           0.05026  0.5516 5.211    2,560         331.6                      0
   0.05                       stop 1.5 ATR · alvo 2R 2016-2026   36432    13.13  0.05414           0.04116  0.5505  5.17    1,972         287.8                      0
   0.05                       stop 1.5 ATR · alvo 5R 2016-2026   36398    13.12  0.05383           0.04086  0.5503 5.164    1,959         277.3                      0
   0.05                       stop 1.5 ATR · alvo 3R 2016-2026   36403    13.12  0.05371           0.04073  0.5504 5.137    1,955         271.3                      0
   0.05    stop 1.5 ATR · metade em 1R + resto em 3R 2016-2026   36410    13.13  0.05284           0.03987  0.5522 5.123    1,924         267.9                      0
   0.05                       stop 1.5 ATR · alvo 1R 2016-2026   36441    13.14  0.05233           0.03936  0.5521  4.94    1,907         281.5                      0
   0.05 stop 1.5 ATR · empate em 1R + trailing 0,25R 2016-2026   36440    13.14  0.05229           0.03932  0.5521 5.023    1,906         263.7                      0
   0.05    stop 1.5 ATR · metade em 1R + trailing 1R 2016-2026   36429    13.13  0.05164           0.03867  0.5521 4.985    1,881           298                      0
   0.05  stop 1.5 ATR · metade em 1R + trailing 0,5R 2016-2026   36439    13.14  0.05117            0.0382  0.5521   4.9    1,865         289.2                      0
   0.05    stop 1.5 ATR · empate em 1R + trailing 1R 2016-2026   36429    13.13  0.05099           0.03802  0.5517 4.963    1,858         306.3                      0
   0.05  stop 1.5 ATR · empate em 1R + trailing 0,5R 2016-2026   36439    13.14  0.04999           0.03702  0.5521 4.838    1,822         287.7                      0
   0.05                         stop 2 ATR · alvo 2R 2016-2026   36322    13.09  0.04029           0.03056  0.5522 4.902    1,464         216.8                      0
   0.05                         stop 2 ATR · alvo 5R 2016-2026   36306    13.09  0.04022           0.03048  0.5522 4.842    1,460         220.7                      0
   0.05                         stop 2 ATR · alvo 3R 2016-2026   36307    13.09  0.03994            0.0302  0.5522 4.824    1,450         220.7                      0
   0.05      stop 2 ATR · metade em 1R + resto em 3R 2016-2026   36307    13.09   0.0399           0.03016  0.5526 4.841    1,448           213                      0
   0.05                         stop 2 ATR · alvo 1R 2016-2026   36341     13.1  0.03981           0.03008  0.5525 4.701    1,447         234.6                      0
   0.05   stop 2 ATR · empate em 1R + trailing 0,25R 2016-2026   36337     13.1  0.03964           0.02991  0.5525 4.742    1,440         230.3                      0
   0.05    stop 2 ATR · metade em 1R + trailing 0,5R 2016-2026   36334     13.1  0.03931           0.02957  0.5525 4.706    1,428         237.7                      0
   0.05      stop 2 ATR · metade em 1R + trailing 1R 2016-2026   36310    13.09  0.03926           0.02952  0.5526 4.782    1,426         224.6                      0
   0.05    stop 2 ATR · empate em 1R + trailing 0,5R 2016-2026   36334     13.1  0.03885           0.02912  0.5525 4.704    1,412         237.1                      0
   0.05      stop 2 ATR · empate em 1R + trailing 1R 2016-2026   36310    13.09  0.03841           0.02867  0.5526 4.754    1,395         228.2                      0
   0.05                       stop 0.5 ATR · alvo 5R 2023-2026   12658    13.27   0.2139            0.1706  0.5103 5.417    2,707           203                      0
   0.05                       stop 0.5 ATR · alvo 3R 2023-2026   12658    13.27      0.2            0.1567  0.5114 5.329    2,532         188.7                      0
   0.05                       stop 0.5 ATR · alvo 2R 2023-2026   12658    13.27   0.1812            0.1379  0.5185 5.245    2,294         183.7                      0
   0.05    stop 0.5 ATR · empate em 1R + trailing 1R 2023-2026   12658    13.27   0.1586            0.1153  0.5697 4.749    2,008         164.3                      0
   0.05  stop 0.5 ATR · empate em 1R + trailing 0,5R 2023-2026   12658    13.27    0.157            0.1136  0.5865 5.237    1,987         182.5                      0
   0.05    stop 0.5 ATR · metade em 1R + resto em 3R 2023-2026   12658    13.27    0.153            0.1097  0.5865  4.94    1,937           167                      0
   0.05 stop 0.5 ATR · empate em 1R + trailing 0,25R 2023-2026   12658    13.27   0.1512            0.1079  0.5865 5.397    1,915           148                      0
   0.05    stop 0.5 ATR · metade em 1R + trailing 1R 2023-2026   12658    13.27   0.1475            0.1042  0.5865 4.962    1,867         168.3                      0
   0.05  stop 0.5 ATR · metade em 1R + trailing 0,5R 2023-2026   12658    13.27   0.1467            0.1033  0.5865  5.17    1,856         179.6                      0
   0.05                       stop 0.5 ATR · alvo 1R 2023-2026   12658    13.27   0.1363           0.09302  0.5865 5.008    1,726         175.9                      0
   0.05                         stop 1 ATR · alvo 2R 2023-2026   12658    13.27   0.1273            0.1057  0.5664 5.097    1,611         100.2                      0
   0.05                         stop 1 ATR · alvo 3R 2023-2026   12658    13.27    0.127            0.1054  0.5664 5.203    1,608         109.4                      0
   0.05                         stop 1 ATR · alvo 5R 2023-2026   12654    13.26   0.1228            0.1011  0.5662 5.238    1,554         109.7                      0
   0.05      stop 1 ATR · metade em 1R + resto em 3R 2023-2026   12658    13.27    0.113           0.09133  0.5753 5.335    1,430         101.6                      0
   0.05   stop 1 ATR · empate em 1R + trailing 0,25R 2023-2026   12658    13.27   0.1112           0.08957  0.5753  5.35    1,408         106.9                      0
   0.05                         stop 1 ATR · alvo 1R 2023-2026   12658    13.27   0.1091           0.08745  0.5753 5.345    1,381         104.4                      0
   0.05      stop 1 ATR · metade em 1R + trailing 1R 2023-2026   12658    13.27   0.1091           0.08743  0.5753 5.226    1,381         97.95                      0
   0.05      stop 1 ATR · empate em 1R + trailing 1R 2023-2026   12658    13.27   0.1091           0.08741  0.5737 4.999    1,381         98.22                      0
   0.05    stop 1 ATR · metade em 1R + trailing 0,5R 2023-2026   12658    13.27   0.1084           0.08676  0.5753 5.278    1,372         104.9                      0
   0.05    stop 1 ATR · empate em 1R + trailing 0,5R 2023-2026   12658    13.27   0.1077           0.08606  0.5753 5.161    1,363         106.5                      0
   0.05                       stop 1.5 ATR · alvo 2R 2023-2026   12517    13.12  0.08935           0.07487  0.5745 5.211    1,118         69.63                      0
   0.05                       stop 1.5 ATR · alvo 3R 2023-2026   12513    13.12  0.08733           0.07286  0.5744 5.231    1,093         71.09                      0
   0.05                       stop 1.5 ATR · alvo 5R 2023-2026   12513    13.12  0.08671           0.07223  0.5744 5.211    1,085         83.33                      0
   0.05                       stop 1.5 ATR · alvo 1R 2023-2026   12520    13.12  0.08661           0.07214  0.5756 5.022    1,084         70.51                      0
   0.05 stop 1.5 ATR · empate em 1R + trailing 0,25R 2023-2026   12520    13.12  0.08653           0.07206  0.5756 4.987    1,083         70.24                      0
   0.05    stop 1.5 ATR · metade em 1R + resto em 3R 2023-2026   12516    13.12  0.08627            0.0718  0.5754 5.205    1,080         70.86                      0
   0.05    stop 1.5 ATR · metade em 1R + trailing 1R 2023-2026   12518    13.12  0.08544           0.07097  0.5755 5.137    1,070         69.22                      0
   0.05  stop 1.5 ATR · metade em 1R + trailing 0,5R 2023-2026   12520    13.12  0.08521           0.07074  0.5756 5.029    1,067         66.69                      0
   0.05    stop 1.5 ATR · empate em 1R + trailing 1R 2023-2026   12518    13.12  0.08427            0.0698  0.5753 5.161    1,055         70.36                      0
   0.05  stop 1.5 ATR · empate em 1R + trailing 0,5R 2023-2026   12520    13.12  0.08381           0.06933  0.5756 5.011    1,049         66.61                      0
   0.05                         stop 2 ATR · alvo 1R 2023-2026   12470    13.07     0.07           0.05914  0.5762 5.136    872.9         48.65                      0
   0.05   stop 2 ATR · empate em 1R + trailing 0,25R 2023-2026   12468    13.07  0.06952           0.05866  0.5761 5.141    866.8         47.64                      0
   0.05    stop 2 ATR · metade em 1R + trailing 0,5R 2023-2026   12468    13.07  0.06855           0.05769  0.5761 5.176    854.7         46.71                      0
   0.05      stop 2 ATR · metade em 1R + resto em 3R 2023-2026   12466    13.07  0.06819           0.05733   0.576 5.258    850.1         50.49                      0
   0.05                         stop 2 ATR · alvo 2R 2023-2026   12467    13.07  0.06796            0.0571  0.5761 5.288    847.2         46.44                      0
   0.05      stop 2 ATR · metade em 1R + trailing 1R 2023-2026   12466    13.07  0.06778           0.05692   0.576 5.261    844.9         50.81                      0
   0.05    stop 2 ATR · empate em 1R + trailing 0,5R 2023-2026   12468    13.07  0.06709           0.05623  0.5761 5.201    836.5         46.44                      0
   0.05                         stop 2 ATR · alvo 3R 2023-2026   12466    13.07  0.06688           0.05602   0.576 5.278    833.8         51.48                      0
   0.05                         stop 2 ATR · alvo 5R 2023-2026   12466    13.07  0.06688           0.05602   0.576 5.278    833.8         52.69                      0
   0.05      stop 2 ATR · empate em 1R + trailing 1R 2023-2026   12466    13.07  0.06559           0.05473   0.576 5.264    817.7         52.02                      0
```

## 3. Dinheiro a partir de US$ 1,000 (sem juros compostos), gestão congelada, teto aberto 5%

Sem um teto para a soma dos stops abertos, a banca quebra US$ 1.000: em dias de rajada o modelo aceita dezenas de entradas no mesmo movimento, e com o volume mínimo isso chegou a 5-90 vezes o saldo em risco ao mesmo tempo (primeira rodada desta varredura). Daqui em diante toda simulação tem teto. `risco`: o stop de cada operação como fração do saldo inicial. `volume_minimo`: quando o menor volume da corretora arrisca mais que isso, `pula` a operação ou entra com o `mínimo`. `teto_aberto`: a soma máxima de stops abertos. `pior_queda_todos_stops`: a pior queda se todo stop aberto batesse junto (a pior queda possível, não a vista). Lucro por ano em US$ sobre os US$ 1.000.

```
  periodo  limiar               gestao  risco volume_minimo  teto_aberto  lucro_ano_usd  lucro_ano_estresse_usd  pior_queda  pior_queda_todos_stops  operacoes  puladas_volume  puladas_teto  posicoes_max  quebrou
2016-2026    0.02 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          720.1                   462.4      0.1902                  0.2276      22558               0        140170            36    False
2016-2026    0.02 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          719.5                   461.4      0.1984                  0.2356      20835               0        141893            17    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          730.7                   473.9      0.1718                  0.2042      19564               0        143164            12    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.01        mínimo         0.05            728                   467.9      0.2068                   0.239      16981               0        145747             9    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          666.3                   415.9      0.2068                  0.2367       8830               0        153898             4    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.001          pula         0.05         -6.399                   -21.6      0.1434                  0.1665       7551          155170             7            61    False
2016-2026    0.02 stop 1 ATR · alvo 3R 0.0025          pula         0.05          31.55                  -19.22      0.2679                  0.2961      11872          149394          1462            28    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.005          pula         0.05          429.6                   231.2      0.3507                  0.3836      18559          128812         15357            16    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.01          pula         0.05          773.6                   497.4      0.1875                  0.2157      18311           34961        109456             9    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.02          pula         0.05          660.6                   409.8      0.2116                  0.2421       8852            3623        150253             4    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          636.9                   485.9      0.2614                  0.3021      13249               0         23486            29    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          645.5                   491.5       0.228                  0.2638      12773               0         23962            21    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          662.9                   505.6      0.2133                  0.2484      12234               0         24501            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          683.8                   515.6       0.228                  0.2611      10819               0         25916             9    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          751.1                   570.3      0.2386                  0.2753       6230               0         30505             4    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.365                  -7.536     0.08408                  0.1115       2280           34455             0            59    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          7.375                  -9.705      0.1607                  0.1921       3597           32863           275            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          234.9                   167.4      0.2004                  0.2409       6134           28422          2179            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          665.9                   502.5      0.1455                   0.178      10514            8889         17332             9    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          748.1                   567.2      0.2232                  0.2474       6232             840         29663             4    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          307.4                   251.3      0.4241                  0.4493       5099               0          7709            23    False
2016-2026    0.08 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          308.9                   250.8      0.4154                  0.4399       4915               0          7893            17    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          317.7                   256.7      0.3924                  0.4204       4745               0          8063            12    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          353.5                   284.1      0.3354                  0.3522       4250               0          8558             9    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          385.6                   305.6      0.3188                  0.3523       2586               0         10222             4    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.001          pula         0.05        -0.9961                  -3.319     0.05904                 0.09028       1019           11789             0            42    False
2016-2026    0.08 stop 1 ATR · alvo 3R 0.0025          pula         0.05           1.25                  -6.417      0.1429                  0.1699       1561           11118           129            27    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.005          pula         0.05          126.3                   98.98      0.1432                  0.1689       2431            9558           819            14    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.01          pula         0.05          316.7                   250.7      0.2283                  0.2445       4022            3246          5540             9    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.02          pula         0.05          381.6                   301.6      0.3328                  0.3677       2585             345          9878             4    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            202                   178.9      0.3916                  0.4385       2150               0          3168            23    False
2016-2026    0.12 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05            201                   177.2      0.3867                   0.435       2076               0          3242            17    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          199.9                   174.9       0.392                   0.438       1984               0          3334            10    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.01        mínimo         0.05            216                   187.3      0.3949                  0.4411       1810               0          3508             8    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          244.4                   210.8      0.3941                  0.4423       1138               0          4180             4    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.001          pula         0.05          -3.75                  -4.559      0.0501                 0.05678        386            4932             0            31    False
2016-2026    0.12 stop 1 ATR · alvo 3R 0.0025          pula         0.05         -11.33                  -14.23       0.204                   0.246        627            4658            33            26    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.005          pula         0.05          40.43                   29.39      0.2127                  0.2552        977            4046           295            12    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.01          pula         0.05          180.8                   154.6      0.2003                  0.2465       1644            1493          2181             8    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.02          pula         0.05          244.9                   211.2      0.3905                  0.4387       1141             160          4017             4    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          137.9                   127.1      0.2325                  0.2786       1020               0          1575            16    False
2016-2026    0.16 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          138.7                   127.6      0.2346                  0.2812        995               0          1600            15    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          137.3                   125.9      0.2305                   0.276        953               0          1642            11    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          154.6                   141.6      0.2418                  0.2836        882               0          1713             8    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          187.1                   171.2      0.2244                  0.2624        580               0          2015             4    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.054                  -2.384     0.03663                 0.04163        176            2419             0            23    False
2016-2026    0.16 stop 1 ATR · alvo 3R 0.0025          pula         0.05         -3.736                  -4.962       0.118                  0.1451        298            2290             7            23    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.005          pula         0.05          14.28                   9.949      0.1265                  0.1646        437            2049           109            12    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.01          pula         0.05          147.2                   135.4      0.1868                  0.2288        803             790          1002             8    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.02          pula         0.05          189.6                   173.7      0.2215                   0.259        581              68          1946             4    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          936.3                   658.1      0.4291                  0.4779       8165               0         59380            18    False
2023-2026    0.02 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          936.3                   657.9      0.4199                  0.4691       7662               0         59883            15    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          942.6                     665      0.3858                  0.4326       7348               0         60197            12    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          1,056                   770.7      0.3123                  0.3523       6168               0         61377             9    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          925.1                   649.3      0.4489                   0.494       2957               0         64588             4    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.958                  -4.464     0.03636                 0.05174       2207           65338             0            37    False
2023-2026    0.02 stop 1 ATR · alvo 3R 0.0025          pula         0.05          87.48                   44.03      0.1327                  0.1675       3660           63646           239            26    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.005          pula         0.05          735.1                   471.1      0.1735                  0.2148       7968           50935          8642            16    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.01          pula         0.05          986.1                   685.3      0.2496                   0.299       6445            7230         53870             9    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.02          pula         0.05          925.1                   649.3      0.4489                   0.494       2957              12         64576             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            878                   713.3      0.1888                  0.2209       4777               0          7881            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          885.9                   718.6      0.1713                   0.211       4655               0          8003            16    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          901.5                   731.8        0.17                  0.1998       4511               0          8147            13    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05            937                   751.2      0.1607                  0.2011       3817               0          8841             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          1,009                   808.9      0.2055                  0.2362       2051               0         10607             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          40.07                   27.97     0.07122                  0.1016        972           11652            34            27    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          419.7                   329.3      0.1388                  0.1801       2708            9013           937            13    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          870.2                   685.8      0.1217                   0.164       3737            1540          7381             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          1,009                   808.9      0.2055                  0.2362       2051               1         10606             4    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          422.9                   358.5      0.1558                  0.2005       1906               0          2493            17    False
2023-2026    0.08 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          418.3                   352.8      0.1548                  0.1996       1858               0          2541            14    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          433.6                   366.4      0.1637                  0.1998       1812               0          2587            12    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          458.9                     381      0.1974                  0.2338       1574               0          2825             9    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          486.7                   396.3       0.165                  0.2045        883               0          3516             4    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.001          pula         0.05          2.925                    1.83     0.01367                 0.01767        231            4168             0            13    False
2023-2026    0.08 stop 1 ATR · alvo 3R 0.0025          pula         0.05          11.44                    6.84     0.07429                  0.1104        437            3952            10            27    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.005          pula         0.05          207.2                   173.1      0.0823                  0.1063       1047            3018           334            13    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.01          pula         0.05          434.3                   357.6      0.1319                  0.1683       1529             467          2403             9    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.02          pula         0.05          486.7                   396.3       0.165                  0.2045        883               0          3516             4    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          269.1                   241.5     0.07037                 0.09143        855               0           941            16    False
2023-2026    0.12 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          264.1                     236     0.06816                 0.08996        843               0           953            17    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          255.8                   226.7     0.06724                  0.1021        814               0           982            10    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          271.5                   236.7     0.09026                  0.1276        722               0          1074             8    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          243.9                   202.9      0.1356                  0.1706        406               0          1390             4    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.001          pula         0.05         0.6616                  0.2127    0.007863                 0.01454        110            1686             0             9    False
2023-2026    0.12 stop 1 ATR · alvo 3R 0.0025          pula         0.05           10.7                   8.654     0.05406                 0.09567        214            1582             0            24    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.005          pula         0.05          87.91                   72.82     0.08149                   0.125        466            1195           135            12    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.01          pula         0.05          245.5                   211.3     0.09616                  0.1344        702             147           947             8    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.02          pula         0.05          243.9                   202.9      0.1356                  0.1706        406               0          1390             4    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            239                     227     0.07063                  0.1001        402               0           450            16    False
2023-2026    0.16 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          242.6                   230.5     0.06798                 0.09799        394               0           458            15    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          241.1                   228.9     0.07429                 0.09931        376               0           476            10    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          277.1                   262.1     0.05955                 0.08867        340               0           512             8    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          337.3                     318     0.06087                  0.0903        200               0           652             3    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.001          pula         0.05         0.6045                  0.3723    0.007693                 0.01083         59             793             0             7    False
2023-2026    0.16 stop 1 ATR · alvo 3R 0.0025          pula         0.05          12.51                   11.47     0.04304                 0.06983        124             728             0            17    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.005          pula         0.05          53.29                   47.53     0.07248                  0.1158        215             572            65            11    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.01          pula         0.05          272.2                   257.2     0.06377                 0.09036        336              73           443             8    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.02          pula         0.05          337.3                     318     0.06087                  0.0903        200               0           652             3    False
```

### O teto aberto, no limiar 0.05

```
  periodo  limiar               gestao  risco volume_minimo  teto_aberto  lucro_ano_usd  lucro_ano_estresse_usd  pior_queda  pior_queda_todos_stops  operacoes  puladas_volume  puladas_teto  posicoes_max  quebrou
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.02          263.6                   192.3       0.212                  0.2285       6941               0         29794            25    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          636.9                   485.9      0.2614                  0.3021      13249               0         23486            29    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo          0.1          913.7                   678.9      0.5168                  0.5671      19859               0         16876            37    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.02          273.2                   200.6       0.189                   0.203       6294               0         30441            10    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          645.5                   491.5       0.228                  0.2638      12773               0         23962            21    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo          0.1          911.3                   671.9      0.5049                  0.5616      19469               0         17266            29    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.02            273                     200      0.1747                  0.1874       5738               0         30997             6    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          662.9                   505.6      0.2133                  0.2484      12234               0         24501            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo          0.1          924.6                   678.7      0.4038                  0.4643      18988               0         17747            23    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.02          290.4                   216.7      0.1531                  0.1689       4797               0         31938             3    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          683.8                   515.6       0.228                  0.2611      10819               0         25916             9    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo          0.1          1,076                   803.4      0.3658                   0.444      17480               0         19255            16    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.02          265.3                   189.9       0.135                  0.1486       2448               0         34287             1    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          751.1                   570.3      0.2386                  0.2753       6230               0         30505             4    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo          0.1          1,385                   1,043      0.3181                  0.3902      11318               0         25417             9    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.02         -1.324                  -6.259     0.06065                 0.07422       2187           34455            93            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.365                  -7.536     0.08408                  0.1115       2280           34455             0            59    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula          0.1         -2.365                  -7.536     0.08408                  0.1115       2280           34455             0            59    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.02          13.81                  0.8633     0.08454                  0.1006       2788           32863          1084            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          7.375                  -9.705      0.1607                  0.1921       3597           32863           275            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula          0.1          8.717                  -9.793      0.2085                  0.2819       3831           32863            41            48    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.02          156.1                     113     0.08803                  0.1044       3888           28422          4425             7    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          234.9                   167.4      0.2004                  0.2409       6134           28422          2179            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula          0.1          258.7                   176.5      0.2784                  0.3516       7410           28422           903            26    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.02          305.2                   231.8     0.07737                 0.08549       4796            8889         23050             3    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          665.9                   502.5      0.1455                   0.178      10514            8889         17332             9    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula          0.1          1,030                   774.4       0.247                  0.3292      16290            8889         11556            17    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.02          265.3                   189.9       0.135                  0.1486       2448             840         33447             1    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          748.1                   567.2      0.2232                  0.2474       6232             840         29663             4    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula          0.1          1,394                   1,051      0.2896                  0.3471      11305             840         24590             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.02          363.1                   283.6       0.115                  0.1329       2459               0         10199            14    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            878                   713.3      0.1888                  0.2209       4777               0          7881            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo          0.1          1,124                   873.1      0.3535                  0.4501       7131               0          5527            28    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.02          375.8                   295.3      0.1144                  0.1322       2294               0         10364            10    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          885.9                   718.6      0.1713                   0.211       4655               0          8003            16    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo          0.1          1,127                   873.1      0.3213                  0.4179       7031               0          5627            24    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.02          374.3                     294       0.111                  0.1288       2137               0         10521             6    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          901.5                   731.8        0.17                  0.1998       4511               0          8147            13    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo          0.1          1,126                   867.1      0.3021                  0.3959       6902               0          5756            23    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.02          414.5                   332.6       0.114                  0.1289       1689               0         10969             3    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05            937                   751.2      0.1607                  0.2011       3817               0          8841             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo          0.1          1,385                   1,087      0.3335                  0.4246       6162               0          6496            16    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.02          327.2                   240.2      0.1325                  0.1491        824               0         11834             1    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          1,009                   808.9      0.2055                  0.2362       2051               0         10607             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo          0.1          1,750                   1,370      0.2869                  0.3724       3713               0          8945             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.02          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula          0.1          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.02          21.99                   11.69     0.06717                 0.07992        820           11652           186            12    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          40.07                   27.97     0.07122                  0.1016        972           11652            34            27    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula          0.1          38.74                   25.88     0.07153                  0.1166       1006           11652             0            40    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.02          255.2                   199.7     0.06954                 0.07988       1623            9013          2022             7    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          419.7                   329.3      0.1388                  0.1801       2708            9013           937            13    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula          0.1          520.2                   409.6      0.1525                  0.1934       3283            9013           362            24    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.02          386.7                   304.7     0.09575                  0.1109       1677            1540          9441             3    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          870.2                   685.8      0.1217                   0.164       3737            1540          7381             9    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula          0.1          1,413                   1,120      0.1759                  0.2595       5960            1540          5158            16    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.02          327.2                   240.2      0.1325                  0.1491        824               1         11833             1    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          1,009                   808.9      0.2055                  0.2362       2051               1         10606             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula          0.1          1,749                   1,370      0.2869                  0.3724       3713               1          8944             9    False
```

## 4. O teste honesto: escolhida em 2016-2022, olhada em 2023-2026

Para cada teto de queda (todos os stops abertos batendo juntos), a combinação (limiar, gestão, risco, volume mínimo, teto aberto) que mais rendeu em 2016-2022 com custo dobrado e sem juros compostos, sem quebrar; ao lado, o que essa mesma combinação fez em 2023-2026, período que a escolha não viu.

```
                    escolha  limiar                                  gestao  risco volume_minimo  teto_aberto  16-22_lucro_ano  16-22_estresse_ano  16-22_queda  23-26_lucro_ano  23-26_estresse_ano  23-26_queda  23-26_quebrou
                queda ≤ 20%    0.05                    stop 1 ATR · alvo 3R   0.01          pula         0.05            558.9               406.5        0.178            870.2               685.8        0.164          False
                queda ≤ 35%    0.05 stop 1 ATR · empate em 1R + trailing 1R   0.02          pula          0.1            1,272               945.9       0.2991            1,674               1,293       0.4977          False
                queda ≤ 50%    0.05 stop 1 ATR · empate em 1R + trailing 1R   0.02          pula          0.1            1,272               945.9       0.2991            1,674               1,293       0.4977          False
a congelada (0,5%, teto 5%)    0.05                    stop 1 ATR · alvo 3R  0.005          pula         0.05            138.1               82.56       0.2409            419.7               329.3       0.1801          False
```

### As 12 melhores de 2016-2022 (queda ≤ 25%), cada uma em 2023-2026

```
 limiar                                     gestao  risco volume_minimo  teto_aberto  16-22_lucro_ano  16-22_estresse_ano  16-22_queda  23-26_lucro_ano  23-26_estresse_ano  23-26_queda
   0.05                       stop 1 ATR · alvo 3R   0.02          pula         0.05            611.2               440.6       0.2474            1,009               808.9       0.2362
   0.05    stop 1 ATR · metade em 1R + resto em 3R   0.02          pula         0.05            609.7               437.5        0.244            961.2               760.1       0.2361
   0.05                       stop 1 ATR · alvo 3R   0.01          pula         0.05            558.9               406.5        0.178            870.2               685.8        0.164
   0.02                       stop 1 ATR · alvo 3R   0.01          pula         0.05            662.3               399.1       0.2157            986.1               685.3        0.299
   0.05                       stop 1 ATR · alvo 2R   0.01          pula         0.05            541.6               388.1       0.1813            850.1               664.6        0.164
   0.05                       stop 1 ATR · alvo 3R  0.005        mínimo         0.05            537.9               387.1       0.2484            901.5               731.8       0.1998
   0.05 stop 2 ATR · empate em 1R + trailing 0,25R   0.02          pula          0.1            533.3               386.3       0.2068            867.9               689.1       0.2066
   0.05    stop 2 ATR · metade em 1R + trailing 1R   0.02          pula          0.1              533               386.2        0.192            781.8               603.5       0.2064
   0.05    stop 1 ATR · empate em 1R + trailing 1R   0.01          pula         0.05              540               385.3       0.1714            851.7               665.6       0.1908
   0.05    stop 2 ATR · empate em 1R + trailing 1R   0.02          pula          0.1            530.4               383.6       0.1926            776.6               598.3       0.2064
   0.05  stop 2 ATR · metade em 1R + trailing 0,5R   0.02          pula          0.1            525.8               378.5       0.2087            739.4               559.6       0.2201
   0.05  stop 2 ATR · empate em 1R + trailing 0,5R   0.02          pula          0.1            522.3                 375       0.2114            730.6               550.9       0.2201
```

