# C13-A: o bid que afunda na virada das 17:00 de Nova York (exploratório)

Os minutos da pesquisa são preço de venda (bid). Logo depois da virada o spread abre e o bid afunda junto, voltando só perto das 18:30. O backtest comprou esse bid afundado com o custo fixo de sempre; quem compra de verdade paga o ask, que subiu o mesmo tanto. Vender (ou fechar uma compra) no bid afundado é o que o mercado faz de fato, então as vendas não têm esse viés.

## O bid contra o fechamento das 16:50, em % do ATR diário (média das noites)

```
mercado   periodo  16:58  17:00  17:05  17:15  17:30  18:00  18:30  19:00
 EURUSD 2016-2022  -0.36  -1.39  -0.57  -0.15  -0.12   0.61   1.36   1.53
 EURUSD 2023-2024  -0.07  -1.44  -1.33  -1.06  -0.52   0.85   1.67   1.18
 EURUSD 2025-2026   0.01  -2.08  -2.12  -1.60  -1.08   0.78   1.04   1.30
 USDJPY 2016-2022  -0.32  -2.00  -1.81  -1.75  -1.59  -1.09  -0.92  -1.22
 USDJPY 2023-2024  -0.51  -2.68  -3.70  -3.81  -3.41  -4.08  -4.74  -3.67
 USDJPY 2025-2026  -1.00  -4.45  -5.55  -4.89  -4.04  -3.26  -2.16  -3.25
 AUDNZD 2016-2022  -1.77  -7.24  -5.60  -4.56  -2.96  -0.83   0.54   0.73
 AUDNZD 2023-2024  -1.56  -7.16 -17.56  -9.13  -5.95  -1.62  -0.24  -0.48
 AUDNZD 2025-2026  -2.10  -6.17 -22.51 -12.01  -8.37  -2.54   0.57  -0.46
 CHFJPY 2016-2022  -1.05  -4.97  -3.84  -4.67  -4.59  -1.03   0.28   0.37
 CHFJPY 2023-2024  -1.00  -3.13 -12.52  -9.95  -7.52  -4.24  -3.08  -2.37
 CHFJPY 2025-2026  -1.81  -6.41 -17.98 -13.39  -9.24  -3.48  -1.24  -1.96
 NZDCHF 2016-2022  -1.55  -5.66  -4.95  -5.41  -4.96  -2.06   0.46   0.31
 NZDCHF 2023-2024  -3.16  -7.40 -23.69 -14.37 -10.59  -3.71   0.27  -0.09
 NZDCHF 2025-2026  -3.09 -10.57 -29.27 -21.26 -15.80  -4.53  -0.24   0.02
 AUDCHF 2016-2022  -1.24  -6.07  -5.18  -5.94  -5.52  -2.65  -0.54  -0.69
 AUDCHF 2023-2024  -1.88  -7.69 -14.59  -8.80  -6.95  -3.11  -1.10  -1.30
 AUDCHF 2025-2026  -2.27 -10.90 -24.74 -14.73 -11.66  -4.60  -1.74  -2.10
 GBPCHF 2016-2022  -0.97  -5.52  -4.91  -5.64  -4.93  -1.69  -0.04  -0.13
 GBPCHF 2023-2024  -0.79  -6.09 -14.48 -11.68  -8.97  -3.67  -1.89  -1.55
 GBPCHF 2025-2026  -1.70  -8.75 -17.66 -16.06 -11.87  -4.65  -2.30  -2.15
 EURJPY 2016-2022  -0.46  -3.28  -2.20  -1.82  -1.48  -0.28   0.43   0.30
 EURJPY 2023-2024  -0.21  -2.23  -6.21  -5.06  -4.07  -3.26  -3.46  -2.69
 EURJPY 2025-2026  -0.92  -2.91  -8.22  -6.58  -5.36  -2.70  -1.48  -2.33
 GBPJPY 2016-2022  -0.43  -3.39  -2.76  -2.76  -2.00  -0.75   0.16  -0.13
 GBPJPY 2023-2024  -0.57  -2.19  -7.67  -6.20  -4.74  -4.23  -4.11  -3.41
 GBPJPY 2025-2026  -1.53  -3.54 -10.68  -9.20  -7.12  -4.28  -2.58  -3.30
 AUDCAD 2016-2022  -1.04  -5.82  -5.00  -4.51  -3.55  -1.67   0.11   0.38
 AUDCAD 2023-2024  -1.06  -8.20 -10.19  -7.02  -5.40  -1.54   0.30   0.50
 AUDCAD 2025-2026  -1.23 -11.41 -15.81 -12.82  -7.82  -2.55  -1.04  -1.03
```

## Operações aceitas pelo walk-forward, por horário de entrada (R por operação)

```
            size                            mean                       
periodo  2016-22           2023-26       2016-22        2023-26        
lado      compra    venda   compra venda  compra  venda  compra   venda
16:45         10        9        9   NaN 0.00493 0.0724   0.162     NaN
17:00         41       52       10     5 -0.0891  0.268   0.585   0.745
17:15   1.32e+03 1.14e+03 2.49e+03   294  0.0224 0.0808   0.265 0.00792
17:30   1.02e+03      986      966   322  0.0116  0.116   0.328 0.00279
17:45        119      280      104    88   0.423 0.0907  0.0611 -0.0509
18:00        330      696      151   281   0.306  0.205   0.098  0.0537
18:15        440 1.66e+03      252   867   0.043 0.0835  0.0917  0.0747
18:30        648 2.62e+03      184   858   0.166   0.12   0.142 0.00897
18:45        220      919       47   254 -0.0291 0.0478   -0.13 -0.0572
19:00        588 2.15e+03      166   727 -0.0365 0.0591  -0.199  0.0577
```

## A vantagem por operação sem as compras de 17:00-18:30

```
  periodo                      regra  trades  mean_net  mean_stress  win_rate     t
2016-2022                como estava   24077   0.05941      0.04115    0.5309 3.308
2016-2022 sem compras 17:00-18:30 NY   20807   0.05875      0.04056    0.5292 3.478
2023-2026                como estava   12658     0.127       0.1054    0.5664 5.203
2023-2026 sem compras 17:00-18:30 NY    8683   0.06701      0.04685    0.5115 2.221
```

## Dinheiro com US$ 1.000 fixos (1% por operação, teto 5%, freio -5%), custo do backtest

```
                  regra   periodo  US$/ano  US$/ano estresse  pior_queda  operacoes
            como estava 2016-2026      633               469       0.148      10534
            como estava 2023-2026      832               642       0.141       3762
            como estava 2016-2022      524               372       0.148       6764
sem compras 17:00-17:30 2016-2026      490               341       0.139       9893
sem compras 17:00-17:30 2023-2026      456               305       0.169       3265
sem compras 17:00-17:30 2016-2022      503               356       0.139       6615
sem compras 17:00-18:30 2016-2026      457               317       0.162       9354
sem compras 17:00-18:30 2023-2026      349               214       0.141       2976
sem compras 17:00-18:30 2016-2022      512               370       0.162       6377
   sem nada 17:00-18:30 2016-2026      350               229       0.184       8007
   sem nada 17:00-18:30 2023-2026      285               173       0.195       2562
   sem nada 17:00-18:30 2016-2022      380               255       0.184       5441
```

## Decisão

O robô ao vivo não abre compra das 17:00 às 18:30 de Nova York e fecha tudo 5 minutos antes da virada (antes, 45 s): a janela e a margem saem da medição do bid, não do dinheiro. A expectativa honesta de 2023-2026 é a linha sem essas compras; a de 2016-2022 quase não muda.
