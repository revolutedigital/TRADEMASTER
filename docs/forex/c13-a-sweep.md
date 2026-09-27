# C13, frente A: varredura exploratória de limiar, gestão e risco (não é teste pré-registrado)

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
2016-2026    0.02 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          3,698                   1,484      0.2421                  0.2773     119982               0         42746           240    False
2016-2026    0.02 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          3,658                   1,370      0.2357                  0.2724     118357               0         44371           240    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          3,671                   1,492      0.2169                  0.2551     117510               0         45218           240    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          4,350                   1,688       0.235                  0.2698     115454               0         47274           240    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          8,313                   2,441      0.2628                  0.2977      98287               0         64441           212    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.001          pula         0.05         -6.326                  -21.49      0.1427                  0.1657       7550          155170             8            60    False
2016-2026    0.02 stop 1 ATR · alvo 3R 0.0025          pula         0.05          33.95                  -16.11      0.2803                  0.3052      12085          149394          1249            31    False
2016-2026    0.02 stop 1 ATR · alvo 3R  0.005          pula         0.05          698.7                   315.4      0.3738                  0.3979      29193          128812          4723            88    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.01          pula         0.05          4,062                   2,116      0.2367                  0.2727     104722           34961         23045           229    False
2016-2026    0.02 stop 1 ATR · alvo 3R   0.02          pula         0.05          8,307                   2,922      0.2623                  0.2974      98191            3623         60914           212    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          1,592                   967.2      0.2465                   0.274      26805               0          9930           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          1,612                   974.4      0.2392                  0.2626      26706               0         10029           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          1,629                   1,015      0.2236                  0.2569      26526               0         10209           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          1,927                   1,232      0.2531                  0.2754      26311               0         10424           114    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          3,741                   2,221      0.2443                  0.2676      24673               0         12062           102    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.292                  -7.386     0.08332                  0.1108       2279           34455             1            58    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          7.258                  -11.29      0.1565                  0.1844       3583           32863           289            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          277.4                   193.4      0.2035                    0.24       7152           28422          1161            41    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          1,679                   1,162      0.1819                  0.2187      23002            8889          4844            99    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          3,697                   2,325      0.2684                  0.2876      24541             840         11354           101    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            578                   436.7      0.3869                  0.4091       7406               0          5402            50    False
2016-2026    0.08 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05            625                   439.6      0.3295                  0.3532       7522               0          5286            53    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          636.6                   463.7      0.3093                  0.3314       7449               0          5359            53    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          754.9                     528      0.2899                  0.3194       7282               0          5526            54    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          1,203                   862.9      0.3194                  0.3525       5979               0          6829            39    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.001          pula         0.05        -0.9961                  -3.319     0.05904                 0.09028       1019           11789             0            42    False
2016-2026    0.08 stop 1 ATR · alvo 3R 0.0025          pula         0.05          1.206                   -7.04      0.1414                  0.1667       1554           11118           136            25    False
2016-2026    0.08 stop 1 ATR · alvo 3R  0.005          pula         0.05          144.5                   99.44      0.1452                   0.171       2667            9558           583            25    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.01          pula         0.05          675.2                   462.3      0.2266                  0.2566       6545            3246          3017            49    False
2016-2026    0.08 stop 1 ATR · alvo 3R   0.02          pula         0.05          1,185                   850.3      0.2962                  0.3313       5932             345          6531            39    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.001        mínimo         0.05            315                   245.9      0.3915                  0.4214       2688               0          2630            34    False
2016-2026    0.12 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          307.9                   240.9      0.3831                  0.4127       2608               0          2710            32    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          300.8                   248.6      0.3772                  0.4054       2550               0          2768            29    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          346.5                   287.2       0.372                  0.4002       2421               0          2897            28    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          576.6                   473.6      0.3948                  0.4244       1974               0          3344            20    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.001          pula         0.05          -3.75                  -4.559      0.0501                 0.05678        386            4932             0            31    False
2016-2026    0.12 stop 1 ATR · alvo 3R 0.0025          pula         0.05         -11.44                  -14.65      0.2006                  0.2406        620            4658            40            23    False
2016-2026    0.12 stop 1 ATR · alvo 3R  0.005          pula         0.05          42.87                   30.02      0.1955                  0.2322        994            4046           278            14    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.01          pula         0.05          283.7                   235.5      0.1831                  0.2209       2164            1493          1661            25    False
2016-2026    0.12 stop 1 ATR · alvo 3R   0.02          pula         0.05          581.6                     480       0.388                  0.4176       1997             160          3161            20    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          185.8                   169.2      0.2164                  0.2551       1266               0          1329            24    False
2016-2026    0.16 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          181.7                   161.6      0.2197                  0.2582       1218               0          1377            23    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          173.5                   162.4      0.2165                  0.2551       1168               0          1427            19    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          212.3                   189.7      0.2212                  0.2593       1123               0          1472            21    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          349.2                   310.8      0.2112                  0.2488        891               0          1704            13    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.054                  -2.384     0.03663                 0.04163        176            2419             0            23    False
2016-2026    0.16 stop 1 ATR · alvo 3R 0.0025          pula         0.05          -4.11                  -5.327      0.1185                  0.1457        297            2290             8            22    False
2016-2026    0.16 stop 1 ATR · alvo 3R  0.005          pula         0.05          14.46                   9.503      0.1269                  0.1607        433            2049           113            12    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.01          pula         0.05            187                   176.7      0.1702                  0.2057        972             790           833            19    False
2016-2026    0.16 stop 1 ATR · alvo 3R   0.02          pula         0.05          341.9                   309.9      0.2205                  0.2578        881              68          1646            13    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          2,738                   1,364      0.3143                  0.3475      18251               0         49294            97    False
2023-2026    0.02 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          2,591                   1,522      0.3415                  0.3723      16939               0         50606            95    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          2,827                   1,503      0.3136                   0.346      18000               0         49545           101    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          3,290                   1,704       0.324                  0.3554      15770               0         51775            88    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          5,051                   2,548      0.3953                  0.4246       9888               0         57657            58    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.958                  -4.464     0.03636                 0.05174       2207           65338             0            37    False
2023-2026    0.02 stop 1 ATR · alvo 3R 0.0025          pula         0.05          89.92                   45.43      0.1317                  0.1615       3702           63646           197            29    False
2023-2026    0.02 stop 1 ATR · alvo 3R  0.005          pula         0.05          1,080                   621.4      0.1615                  0.1965      11996           50935          4614            55    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.01          pula         0.05          3,287                   815.2       0.302                  0.3362      16063            7230         44252            88    False
2023-2026    0.02 stop 1 ATR · alvo 3R   0.02          pula         0.05          5,051                   2,548      0.3953                  0.4246       9888              12         57645            58    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          1,301                   959.1      0.2173                  0.2538       6892               0          5766            50    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          1,293                   1,001      0.2247                  0.2602       6773               0          5885            49    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          1,266                   995.5      0.2304                  0.2657       6704               0          5954            49    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          1,696                   1,237      0.1823                   0.221       6467               0          6191            45    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          2,849                   2,264      0.2416                  0.2769       4616               0          8042            31    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          41.03                   28.08     0.07101                  0.1016        979           11652            27            29    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          490.5                   379.5      0.1442                  0.1853       3183            9013           462            31    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          1,831                   1,234      0.1528                  0.1948       6797            1540          4321            48    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          2,834                   2,264      0.2416                  0.2769       4597               1          8060            31    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          653.5                   539.8       0.161                  0.1964       2430               0          1969            32    False
2023-2026    0.08 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05            644                   541.7       0.173                  0.2054       2374               0          2025            32    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          663.8                   541.2        0.17                  0.2017       2369               0          2030            32    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          775.9                   570.4      0.2073                  0.2372       2094               0          2305            25    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          1,029                   700.4      0.2045                    0.24       1343               0          3056            14    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.001          pula         0.05          2.925                    1.83     0.01367                 0.01767        231            4168             0            13    False
2023-2026    0.08 stop 1 ATR · alvo 3R 0.0025          pula         0.05          11.21                    6.61     0.07435                  0.1089        436            3952            11            26    False
2023-2026    0.08 stop 1 ATR · alvo 3R  0.005          pula         0.05          209.2                   171.6      0.1019                  0.1312       1115            3018           266            19    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.01          pula         0.05          754.5                   553.4      0.1318                  0.1642       2043             467          1889            24    False
2023-2026    0.08 stop 1 ATR · alvo 3R   0.02          pula         0.05          1,029                   700.4      0.2045                    0.24       1343               0          3056            14    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          338.8                   297.6      0.1197                  0.1457       1011               0           785            22    False
2023-2026    0.12 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          339.6                   297.7       0.123                  0.1548       1000               0           796            22    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          314.2                   274.4      0.1156                  0.1565        948               0           848            21    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          375.5                   323.4     0.08782                  0.1252        867               0           929            15    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          434.1                   320.4      0.1341                  0.1688        529               0          1267             7    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.001          pula         0.05         0.6616                  0.2127    0.007863                 0.01454        110            1686             0             9    False
2023-2026    0.12 stop 1 ATR · alvo 3R 0.0025          pula         0.05           10.7                   8.654     0.05406                 0.09567        214            1582             0            24    False
2023-2026    0.12 stop 1 ATR · alvo 3R  0.005          pula         0.05          88.35                   72.01     0.08274                  0.1265        473            1195           128            13    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.01          pula         0.05          327.8                   280.9      0.1278                  0.1617        830             147           819            14    False
2023-2026    0.12 stop 1 ATR · alvo 3R   0.02          pula         0.05          434.1                   320.4      0.1341                  0.1688        529               0          1267             7    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          285.8                   268.2     0.07471                    0.12        485               0           367            18    False
2023-2026    0.16 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          291.6                   276.4     0.07065                  0.1169        469               0           383            18    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          291.9                   272.8      0.0865                  0.1263        451               0           401            16    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          331.1                     307     0.07867                  0.1246        424               0           428            13    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          484.6                     456     0.08312                  0.1262        292               0           560             8    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.001          pula         0.05         0.6045                  0.3723    0.007693                 0.01083         59             793             0             7    False
2023-2026    0.16 stop 1 ATR · alvo 3R 0.0025          pula         0.05          12.51                   11.47     0.04304                 0.06983        124             728             0            17    False
2023-2026    0.16 stop 1 ATR · alvo 3R  0.005          pula         0.05          52.68                   45.71      0.0772                  0.1205        220             572            60            12    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.01          pula         0.05          331.9                   311.9     0.07596                  0.1209        416              73           363            14    False
2023-2026    0.16 stop 1 ATR · alvo 3R   0.02          pula         0.05          484.6                     456     0.08312                  0.1262        292               0           560             8    False
```

### O teto aberto, no limiar 0.05

```
  periodo  limiar               gestao  risco volume_minimo  teto_aberto  lucro_ano_usd  lucro_ano_estresse_usd  pior_queda  pior_queda_todos_stops  operacoes  puladas_volume  puladas_teto  posicoes_max  quebrou
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.02          606.8                   392.1       0.212                  0.2244      12590               0         24145            30    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          1,592                   967.2      0.2465                   0.274      26805               0          9930           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo          0.1          1,885                   1,452      0.5165                  0.5567      30862               0          5873           176    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.02          615.4                   441.3      0.1939                  0.2041      12297               0         24438            30    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          1,612                   974.4      0.2392                  0.2626      26706               0         10029           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo          0.1          1,850                   1,442      0.4999                  0.5368      30496               0          6239           171    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.02          663.1                   433.2      0.1895                  0.2034      12636               0         24099            32    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          1,629                   1,015      0.2236                  0.2569      26526               0         10209           118    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo          0.1          1,886                   1,466      0.4403                  0.4926      30567               0          6168           168    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.02          799.9                   489.8      0.2101                  0.2191      11594               0         25141            24    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          1,927                   1,232      0.2531                  0.2754      26311               0         10424           114    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo          0.1          2,179                   1,672      0.3398                   0.401      30251               0          6484           159    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.02          968.5                   417.9      0.2623                  0.2763       6332               0         30403            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          3,741                   2,221      0.2443                  0.2676      24673               0         12062           102    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo          0.1          4,477                   3,416      0.3481                  0.3997      29349               0          7386           154    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.02         -1.344                   -6.21     0.06089                 0.07446       2185           34455            95            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05         -2.292                  -7.386     0.08332                  0.1108       2279           34455             1            58    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula          0.1         -2.365                  -7.536     0.08408                  0.1115       2280           34455             0            59    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.02          12.54                  0.2443     0.08996                  0.1064       2821           32863          1051            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          7.258                  -11.29      0.1565                  0.1844       3583           32863           289            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula          0.1          7.817                  -11.18      0.2098                  0.2796       3821           32863            51            46    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.02            235                   153.9     0.09305                  0.1099       5317           28422          2996            16    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          277.4                   193.4      0.2035                    0.24       7152           28422          1161            41    False
2016-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula          0.1          285.7                   204.4      0.2776                  0.3465       7807           28422           506            67    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.02          918.9                   580.5      0.1008                  0.1183      13055            8889         14791            27    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          1,679                   1,162      0.1819                  0.2187      23002            8889          4844            99    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula          0.1          1,878                   1,366      0.2752                   0.342      25694            8889          2152           144    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.02          968.5                   417.9      0.2623                  0.2763       6334             840         29561            13    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          3,697                   2,325      0.2684                  0.2876      24541             840         11354           101    False
2016-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula          0.1          4,502                   3,415      0.3305                    0.39      29929             840          5966           156    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.02          769.9                   490.2        0.12                  0.1374       3494               0          9164            16    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo         0.05          1,301                   959.1      0.2173                  0.2538       6892               0          5766            50    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001        mínimo          0.1          1,777                   1,361      0.3562                  0.4147       9593               0          3065            94    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.02          814.7                   491.7      0.1268                  0.1439       3461               0          9197            17    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo         0.05          1,293                   1,001      0.2247                  0.2602       6773               0          5885            49    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025        mínimo          0.1          1,765                   1,352      0.3451                  0.4036       9490               0          3168            94    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.02          817.7                   557.9      0.1296                   0.146       3373               0          9285            17    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo         0.05          1,266                   995.5      0.2304                  0.2657       6704               0          5954            49    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005        mínimo          0.1          1,842                   1,384      0.2757                  0.3479       9696               0          2962            94    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.02          903.7                   701.9      0.1202                  0.1344       2815               0          9843            12    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo         0.05          1,696                   1,237      0.1823                   0.221       6467               0          6191            45    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01        mínimo          0.1          2,346                   1,829      0.2953                  0.3637       9537               0          3121            94    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.02          685.9                   341.4      0.1311                  0.1453       1199               0         11459             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo         0.05          2,849                   2,264      0.2416                  0.2769       4616               0          8042            31    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02        mínimo          0.1          4,962                   3,526      0.3032                  0.3659       8399               0          4259            80    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.02          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula         0.05          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.001          pula          0.1          6.007                   2.706     0.02184                 0.02728        564           12094             0            18    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.02          23.64                   13.04     0.06739                 0.08009        827           11652           179            12    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula         0.05          41.03                   28.08     0.07101                  0.1016        979           11652            27            29    False
2023-2026    0.05 stop 1 ATR · alvo 3R 0.0025          pula          0.1          38.74                   25.88     0.07153                  0.1166       1006           11652             0            40    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.02          349.6                   265.8     0.07177                 0.08213       2114            9013          1531            11    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula         0.05          490.5                   379.5      0.1442                  0.1853       3183            9013           462            31    False
2023-2026    0.05 stop 1 ATR · alvo 3R  0.005          pula          0.1          584.6                   453.5      0.1525                  0.1934       3544            9013           101            49    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.02          736.4                   585.8      0.1115                  0.1275       2613            1540          8505            10    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula         0.05          1,831                   1,234      0.1528                  0.1948       6797            1540          4321            48    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.01          pula          0.1          2,590                   1,871      0.1862                   0.262       9669            1540          1449            97    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.02          685.9                   341.4      0.1311                  0.1453       1199               1         11458             4    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula         0.05          2,834                   2,264      0.2416                  0.2769       4597               1          8060            31    False
2023-2026    0.05 stop 1 ATR · alvo 3R   0.02          pula          0.1          4,957                   3,526      0.3032                  0.3659       8384               1          4273            80    False
```

## 4. O teste honesto: escolhida em 2016-2022, olhada em 2023-2026

Para cada teto de queda (todos os stops abertos batendo juntos), a combinação (limiar, gestão, risco, volume mínimo, teto aberto) que mais rendeu em 2016-2022 com custo dobrado e sem juros compostos, sem quebrar; ao lado, o que essa mesma combinação fez em 2023-2026, período que a escolha não viu.

```
                    escolha  limiar                                  gestao  risco volume_minimo  teto_aberto  16-22_lucro_ano  16-22_estresse_ano  16-22_queda  23-26_lucro_ano  23-26_estresse_ano  23-26_queda  23-26_quebrou
                queda ≤ 20%    0.05 stop 1 ATR · metade em 1R + resto em 3R   0.01          pula         0.05            947.3               579.2       0.1899            1,653               1,102       0.1967          False
                queda ≤ 35%    0.02                    stop 1 ATR · alvo 3R   0.01          pula          0.1            3,232               1,727       0.3317            4,880               2,336       0.4909          False
                queda ≤ 50%    0.02                    stop 1 ATR · alvo 3R   0.02          pula          0.1            7,562               2,875       0.4292            9,869               5,308       0.6013          False
a congelada (0,5%, teto 5%)    0.05                    stop 1 ATR · alvo 3R  0.005          pula         0.05            131.6               77.48         0.24            490.5               379.5       0.1853          False
```

### As 12 melhores de 2016-2022 (queda ≤ 25%), cada uma em 2023-2026

```
 limiar                                  gestao  risco volume_minimo  teto_aberto  16-22_lucro_ano  16-22_estresse_ano  16-22_queda  23-26_lucro_ano  23-26_estresse_ano  23-26_queda
   0.05                    stop 2 ATR · alvo 1R   0.02          pula          0.1            959.8               667.7       0.2278            1,761               1,210       0.2231
   0.05                  stop 1.5 ATR · alvo 3R   0.02          pula         0.05            1,099               658.6       0.2063            1,510               1,133       0.1989
   0.05                    stop 1 ATR · alvo 2R   0.01          pula         0.05            1,010               653.4       0.2102            1,770               1,155       0.1947
   0.05 stop 2 ATR · metade em 1R + trailing 1R   0.02          pula          0.1            963.3               643.7       0.2262            1,707               1,152       0.2297
   0.05                    stop 1 ATR · alvo 3R   0.01          pula         0.05            997.7               643.5       0.2181            1,831               1,234       0.1948
   0.05 stop 2 ATR · metade em 1R + resto em 3R   0.02          pula          0.1            969.5               641.2       0.2295            1,713               1,089       0.2491
   0.05 stop 2 ATR · empate em 1R + trailing 1R   0.02          pula          0.1            943.1               639.3       0.2258            1,732               1,125        0.228
   0.05 stop 1 ATR · empate em 1R + trailing 1R   0.01          pula         0.05              918               612.3       0.2133            1,639               1,101       0.1972
   0.05                  stop 1.5 ATR · alvo 2R   0.02          pula         0.05            1,073                 604       0.2043            1,576               1,135       0.2028
   0.05                    stop 2 ATR · alvo 2R   0.02          pula          0.1            942.4               602.7        0.247            1,705               1,168       0.2527
   0.05                    stop 1 ATR · alvo 5R   0.01          pula         0.05            940.9               597.1       0.2234            1,847               1,270       0.2026
   0.05                  stop 1.5 ATR · alvo 3R   0.02        mínimo         0.05            973.8               596.2       0.2196            1,379               1,023       0.2076
```

