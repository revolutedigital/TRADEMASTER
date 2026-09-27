# C13-A e Gotobi na mesma conta (exploratório, não é teste pré-registrado)

Conta de US$ 1.000, sem juros compostos (tamanho e tetos sempre sobre os US$ 1.000). C13-A com as regras que rodam na demo (1% por operação, stops abertos até 5%, freio a -5% do dia). Cada perna do Gotobi (configuração B, USDJPY) arrisca a fração indicada dos US$ 1.000 por ATR diário, em lotes inteiros de 1.000 unidades e nunca menos que um (o que o robô faz); `lote mínimo (hoje)` é o que roda hoje. Lucro por ano em US$; `queda`: a pior queda do saldo realizado da conta, com as duas juntas.

Escolhido só com 2016-2022 (maior lucro com custo dobrado e queda ≤ 25%): **5%**.

```
            gotobi   periodo  lucro_ano  queda  lucro_ano_estresse  queda_estresse  gotobi_sozinho_ano  gotobi_sozinho_queda
                1% 2016-2022      568.1 0.1351               399.9          0.1761               44.25               0.02609
                2% 2016-2022        607 0.1365               420.3          0.1698               83.15               0.03745
                3% 2016-2022      662.9 0.1419               455.4          0.1602               139.1               0.05152
                4% 2016-2022      717.6 0.1434                 489          0.1596               193.8               0.07623
                5% 2016-2022      771.3 0.1512               521.7          0.1694               247.4                0.1005
         desligado 2016-2022      523.9 0.1472               371.6          0.1986                 NaN                   NaN
lote mínimo (hoje) 2016-2022      565.6 0.1351               399.3          0.1761               41.74               0.02609
                1% 2016-2026      667.3 0.1351               487.7          0.1764               34.43               0.02609
                2% 2016-2026      696.4 0.1365                 500          0.1701               63.51               0.03745
                3% 2016-2026      740.9 0.1419               524.3          0.1605                 108               0.05152
                4% 2016-2026      781.6 0.1434                 545          0.1596               148.7               0.07623
                5% 2016-2026      824.7 0.1512               567.9          0.1694               191.8                0.1005
         desligado 2016-2026      632.9 0.1484               469.2          0.1986                 NaN                   NaN
lote mínimo (hoje) 2016-2026      665.4 0.1351               487.3          0.1764               32.51               0.02609
                1% 2023-2026        848 0.1465               642.2          0.2182                15.7               0.02502
                2% 2023-2026      858.3 0.1544               638.9          0.2145               26.02               0.04354
                3% 2023-2026      880.9 0.1644               642.6          0.2101               48.67               0.07372
                4% 2023-2026      894.9 0.1761               638.5          0.2081               62.68               0.09287
                5% 2023-2026      917.8 0.1836               642.8          0.2147               85.58                 0.101
         desligado 2023-2026      832.3 0.1414               642.3          0.2259                 NaN                   NaN
lote mínimo (hoje) 2023-2026      847.2 0.1465                 642          0.2182               14.89               0.02537
```

