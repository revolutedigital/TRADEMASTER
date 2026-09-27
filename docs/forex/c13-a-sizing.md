# C13, frente A: regras de dinheiro (exploratório, não é teste pré-registrado)

Entradas e saídas congeladas (as escolhas do walk-forward, stop 1 ATR, alvo 3R, fim do dia FX); só muda quanto cada operação arrisca e quando uma operação é recusada. Conta de US$ 1.000, sem juros compostos, volume mínimo real de cada mercado na demo; lucro por ano em US$ sobre os US$ 1.000; `queda_todos_stops`: a pior queda se todo stop aberto batesse junto. `teto_moeda`: a soma máxima de stops abertos numa mesma moeda e direção (inf = sem teto). `freio_dia`: sem novas entradas no resto do dia FX depois de perder essa fração no dia (inf = sem freio; as posições abertas não são fechadas).

**Isto não aprova nada.** As regras são escolhidas só com 2016-2022 e lidas em 2023-2026.

## 1. Uma regra de cada vez, a partir da de hoje (1% em tudo (a de hoje), teto aberto 5%)

```
      regra                                valor   periodo  lucro_ano_usd  estresse  queda_todos_stops  operacoes
    esquema               1% em tudo (a de hoje) 2016-2022          960.6     662.3             0.2478      11983
    esquema                         1,5% em tudo 2016-2022          1,517     901.3             0.3283      12983
    esquema                           2% em tudo 2016-2022          2,024     736.8             0.3536      12132
    esquema        1% / 1,5% / 2% pela confiança 2016-2022          1,224     772.8             0.2965      11485
    esquema          1% / 2% / 3% pela confiança 2016-2022          1,299     815.2             0.2936      10830
    esquema       0,5% abaixo de 0,05 + 1% acima 2016-2022          1,237     684.4             0.2576      22699
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2016-2022          1,425     761.2             0.2561      21268
    esquema               1% em tudo (a de hoje) 2023-2026          1,772     1,114             0.2248       6601
    esquema                         1,5% em tudo 2023-2026          2,183     1,415              0.258       5456
    esquema                           2% em tudo 2023-2026          2,637     1,975             0.3572       4284
    esquema        1% / 1,5% / 2% pela confiança 2023-2026          1,964     1,236             0.2286       5954
    esquema          1% / 2% / 3% pela confiança 2023-2026          2,230     1,523             0.2231       5702
    esquema       0,5% abaixo de 0,05 + 1% acima 2023-2026          2,147     1,125             0.2636      13675
    esquema 0,5% / 1% / 1,5% / 2% pela confiança 2023-2026          2,497     1,251             0.2827      13367
  freio_dia                                  inf 2016-2022          960.6     662.3             0.2478      11983
  freio_dia                                 0.05 2016-2022            898     653.8             0.2502      11748
  freio_dia                                 0.03 2016-2022          940.3     615.5             0.2233      11944
  freio_dia                                 0.02 2016-2022          936.9     606.8             0.2004      11882
  freio_dia                                  inf 2023-2026          1,772     1,114             0.2248       6601
  freio_dia                                 0.05 2023-2026          1,720     1,041             0.2023       6490
  freio_dia                                 0.03 2023-2026          1,736     1,079             0.1989       6519
  freio_dia                                 0.02 2023-2026          1,732     1,081             0.1983       6492
teto_aberto                                 0.05 2016-2022          960.6     662.3             0.2478      11983
teto_aberto                                0.075 2016-2022          1,007     700.3             0.3123      13529
teto_aberto                                  0.1 2016-2022          1,149       696             0.3325      14546
teto_aberto                                 0.05 2023-2026          1,772     1,114             0.2248       6601
teto_aberto                                0.075 2023-2026          2,302     1,620             0.2342       8637
teto_aberto                                  0.1 2023-2026          2,547     1,912              0.262       9655
 teto_moeda                                  inf 2016-2022          960.6     662.3             0.2478      11983
 teto_moeda                                 0.03 2016-2022          914.8     598.1             0.2102      11429
 teto_moeda                                 0.02 2016-2022          839.4     533.6             0.2013      10695
 teto_moeda                                  inf 2023-2026          1,772     1,114             0.2248       6601
 teto_moeda                                 0.03 2023-2026          1,635     1,092             0.2404       6334
 teto_moeda                                 0.02 2023-2026          1,493     1,049             0.2313       5762
```

## 2. As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026

```
  posição                       esquema  teto_aberto  teto_moeda  freio_dia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_queda
       1ª   1% / 2% / 3% pela confiança        0.075        0.03       0.05      1,584           909.3       0.2366      2,433           1,556       0.2299
       2ª   1% / 2% / 3% pela confiança        0.075        0.03       0.03      1,521           902.8       0.2283      2,452           1,555       0.2547
       3ª 1% / 1,5% / 2% pela confiança        0.075        0.03       0.03      1,329           850.3       0.2257      2,337           1,507        0.237
       4ª 1% / 1,5% / 2% pela confiança        0.075        0.03       0.05      1,359           815.9       0.2406      2,232           1,585       0.2391
       5ª   1% / 2% / 3% pela confiança        0.075        0.03       0.02      1,397           808.7       0.2336      2,331           1,557       0.2467
       6ª   1% / 2% / 3% pela confiança         0.05        0.03       0.03      1,173           790.8       0.2058      1,811           1,317       0.2805
       7ª   1% / 2% / 3% pela confiança         0.05        0.03       0.05      1,261           787.8       0.1983      1,826           1,312       0.2301
       8ª                  1,5% em tudo         0.05        0.03       0.03      1,310           786.2       0.2206      2,132           1,430       0.2327
       9ª   1% / 2% / 3% pela confiança        0.075        0.02       0.03      1,227           768.5       0.2187      1,816           1,141       0.1968
      10ª   1% / 2% / 3% pela confiança         0.05        0.03       0.02      1,232           760.4       0.2004      1,705           1,295       0.2684
      11ª                  1,5% em tudo         0.05        0.03       0.02      1,258           748.6       0.2262      2,132           1,411       0.2453
      12ª   1% / 2% / 3% pela confiança         0.05        0.03        inf      1,277           746.7       0.1987      1,853           1,312       0.2461
a de hoje        1% em tudo (a de hoje)         0.05         inf        inf      960.6           662.3       0.2478      1,772           1,114       0.2248
```

