# C13-A e Gotobi na mesma conta (exploratório, não é teste pré-registrado)

Conta de US$ 1.000, sem juros compostos. C13-A com as regras que rodam na demo (1% / 2% / 3% pela confiança, stops abertos até 7,5%, 3% por moeda, freio a -5% do dia, prioridade pela confiança). Cada perna do Gotobi (configuração B, USDJPY) arrisca a fração indicada dos US$ 1.000 por ATR diário, em lotes inteiros de 1.000 unidades e nunca menos que um (o que o robô faz); `lote mínimo (hoje)` é o que roda hoje. Lucro por ano em US$; `queda`: a pior queda do saldo realizado da conta, com as duas juntas.

Escolhido só com 2016-2022 (maior lucro com custo dobrado e queda ≤ 25%): **5%**.

```
            gotobi   periodo  lucro_ano  queda  lucro_ano_estresse  queda_estresse  gotobi_sozinho_ano  gotobi_sozinho_queda
                1% 2016-2022      1,812 0.2008               984.2          0.2161               44.25               0.02609
                2% 2016-2022      1,850 0.2005               1,005          0.2161               83.15               0.03745
                3% 2016-2022      1,906  0.207               1,040           0.225               139.1               0.05152
                4% 2016-2022      1,961 0.2082               1,073          0.2283               193.8               0.07623
                5% 2016-2022      2,015 0.2155               1,106          0.2405               247.4                0.1005
         desligado 2016-2022      1,767 0.2127                 956           0.224                 NaN                   NaN
lote mínimo (hoje) 2016-2022      1,809 0.2008               983.7          0.2161               41.74               0.02609
                1% 2016-2026      2,952 0.2007               1,857          0.2139               34.43               0.02609
                2% 2016-2026      2,981 0.2005               1,870          0.2161               63.51               0.03745
                3% 2016-2026      3,026  0.207               1,894           0.225                 108               0.05152
                4% 2016-2026      3,066 0.2082               1,915          0.2283               148.7               0.07623
                5% 2016-2026      3,109 0.2155               1,938          0.2405               191.8                0.1005
         desligado 2016-2026      2,918 0.2122               1,839           0.224                 NaN                   NaN
lote mínimo (hoje) 2016-2026      2,950 0.2007               1,857          0.2139               32.51               0.02609
                1% 2023-2026      2,827  0.203               1,696          0.2387                15.7               0.02502
                2% 2023-2026      2,837 0.2108               1,693          0.2532               26.02               0.04354
                3% 2023-2026      2,860 0.2168               1,697           0.268               48.67               0.07372
                4% 2023-2026      2,874 0.2246               1,693          0.2829               62.68               0.09287
                5% 2023-2026      2,897 0.2282               1,697           0.293               85.58                 0.101
         desligado 2023-2026      2,811 0.1989               1,696          0.2445                 NaN                   NaN
lote mínimo (hoje) 2023-2026      2,826  0.203               1,696          0.2386               14.89               0.02537
```

