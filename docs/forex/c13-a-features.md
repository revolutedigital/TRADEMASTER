# C13, frente A: informações novas para o modelo (exploratório, não é teste pré-registrado)

Cada grupo de informações é somado às do modelo congelado, e o mesmo modelo (profundidade 3, peso 2000) é retreinado pelo walk-forward da descoberta: cada ano decidido por um modelo que só viu operações encerradas antes dele. Operações aceitas com previsão acima de 0.05. Julga-se em 2016-2022 e lê-se em 2023-2026.

Grupos: `{"rajada": ["burst_all", "burst_market", "burst_set"], "dolar": ["usd_side", "usd_breadth", "usd_count", "usd_aligned"], "rajada+dolar": ["burst_all", "burst_market", "burst_set", "usd_side", "usd_breadth", "usd_count", "usd_aligned"], "volatilidade": ["sigma_left", "vol_5m_surprise", "vol_day_surprise", "move_z"], "vix": ["vix_level", "vix_ratio", "vix_change"]}`

```
       grupo   periodo  trades  mean_net  mean_stress  win_rate     t  market_share  trades_per_day
modelo atual 2016-2022   24077   0.05941      0.04115    0.5309 3.308        0.8378           13.23
modelo atual 2023-2026   12658     0.127       0.1054    0.5664 5.203        0.8919           13.27
      rajada 2016-2022   23859   0.06818      0.04964    0.5334  3.66        0.8649           13.11
      rajada 2023-2026   12459    0.1224       0.1011    0.5642  4.82        0.9459           13.06
       dolar 2016-2022   24606   0.06158      0.04327    0.5282 3.627        0.8378           13.52
       dolar 2023-2026   12690    0.1258       0.1045     0.565 5.357        0.9189            13.3
rajada+dolar 2016-2022   24896   0.05995      0.04152    0.5283  3.48        0.8108           13.68
rajada+dolar 2023-2026   11941    0.1231       0.1015    0.5694 5.024        0.9189           12.52
volatilidade 2016-2022   25601   0.06122      0.04292    0.5256 3.325        0.8108           14.07
volatilidade 2023-2026   11682    0.1273       0.1064     0.555 4.308        0.8649           12.25
         vix 2016-2022   36187   0.01823   -0.0004526    0.5068 1.198        0.7027           19.88
         vix 2023-2026   11775   0.04619       0.0254    0.5186 1.068        0.7297           12.34
```
