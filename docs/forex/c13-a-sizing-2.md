# C13, frente A: regras de dinheiro, rodada 2 (exploratório, não é teste pré-registrado)

Em volta das regras que rodam desde 2026-09-27 (1% / 2% / 3% pela confiança, stops abertos até 7,5%, 3% por moeda, freio a -5% do dia). Duas regras novas: `prioridade` — operações que entram no mesmo minuto recebem a vaga do teto em ordem da previsão do modelo, maior primeiro (sem ela, na ordem em que chegam); `tolerancia` — quando o volume mínimo da corretora arrisca mais que o orçamento da operação, entra com o mínimo se ele arriscar até essa vez o orçamento (1 = pula). Mesmas medidas da rodada 1 (docs/forex/c13-a-sizing.md).

**Isto não aprova nada.** Escolhidas só com 2016-2022 e lidas em 2023-2026.

## 1. Uma regra de cada vez, a partir das de hoje

```
      regra                         valor   periodo  lucro_ano_usd  estresse  queda_todos_stops  operacoes
    esquema   1% / 2% / 3% pela confiança 2016-2022          1,584     909.3             0.2366      11866
    esquema 1% / 1,5% / 2% pela confiança 2016-2022          1,359     815.9             0.2406      12454
    esquema   1% / 2% / 3% pela confiança 2023-2026          2,433     1,556             0.2299       6327
    esquema 1% / 1,5% / 2% pela confiança 2023-2026          2,232     1,585             0.2391       6805
  freio_dia                           inf 2016-2022          1,613     814.6             0.2668      11882
  freio_dia                          0.05 2016-2022          1,584     909.3             0.2366      11866
  freio_dia                           inf 2023-2026          2,529     1,560             0.2413       6439
  freio_dia                          0.05 2023-2026          2,433     1,556             0.2299       6327
 prioridade                         False 2016-2022          1,584     909.3             0.2366      11866
 prioridade                          True 2016-2022          1,767       956             0.2476      12073
 prioridade                         False 2023-2026          2,433     1,556             0.2299       6327
 prioridade                          True 2023-2026          2,811     1,696             0.2503       6314
teto_aberto                          0.05 2016-2022          1,261     787.8             0.1983      10688
teto_aberto                         0.075 2016-2022          1,584     909.3             0.2366      11866
teto_aberto                           0.1 2016-2022          1,657     1,067             0.2713      12304
teto_aberto                          0.05 2023-2026          1,826     1,312             0.2301       5226
teto_aberto                         0.075 2023-2026          2,433     1,556             0.2299       6327
teto_aberto                           0.1 2023-2026          2,823     1,683             0.2868       6925
 teto_moeda                           inf 2016-2022          1,700     1,009             0.3794      12504
 teto_moeda                          0.03 2016-2022          1,584     909.3             0.2366      11866
 teto_moeda                          0.02 2016-2022          1,183       744             0.2527      10783
 teto_moeda                           inf 2023-2026          3,219     2,069             0.3108       7315
 teto_moeda                          0.03 2023-2026          2,433     1,556             0.2299       6327
 teto_moeda                          0.02 2023-2026          1,876     1,148             0.1968       5321
 tolerancia                             1 2016-2022          1,584     909.3             0.2366      11866
 tolerancia                           1.5 2016-2022          1,418     841.2             0.3083      13030
 tolerancia                             2 2016-2022          1,446       815             0.2931      13214
 tolerancia                             1 2023-2026          2,433     1,556             0.2299       6327
 tolerancia                           1.5 2023-2026          2,440     1,557             0.2515       6425
 tolerancia                             2 2023-2026          2,422     1,558             0.2692       6398
```

## 2. As 12 melhores de 2016-2022 (custo dobrado, queda ≤ 25%), cada uma em 2023-2026; referência = as de hoje

```
   posição                       esquema  teto_aberto  teto_moeda  freio_dia  prioridade  tolerancia  16-22_ano  16-22_estresse  16-22_queda  23-26_ano  23-26_estresse  23-26_queda
        1ª   1% / 2% / 3% pela confiança        0.075        0.03       0.05        True           1      1,767             956       0.2476      2,811           1,696       0.2503
        2ª   1% / 2% / 3% pela confiança        0.075        0.03       0.05       False           1      1,584           909.3       0.2366      2,433           1,556       0.2299
        3ª 1% / 1,5% / 2% pela confiança        0.075        0.03       0.05        True           1      1,459           882.7       0.2455      2,407           1,745       0.2732
        4ª 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05        True           2      1,462           869.4       0.2231      1,785           1,360       0.2329
        5ª   1% / 2% / 3% pela confiança         0.05        0.03       0.05        True           1      1,261           819.2       0.1977      2,161           1,546          0.2
        6ª 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05        True         1.5      1,286           816.2       0.2248      1,777           1,357       0.2304
        7ª 1% / 1,5% / 2% pela confiança        0.075        0.03       0.05       False           1      1,359           815.9       0.2406      2,232           1,585       0.2391
        8ª   1% / 2% / 3% pela confiança         0.05        0.03        inf        True           1      1,275             803       0.2005      2,226           1,639       0.2137
        9ª 1% / 1,5% / 2% pela confiança         0.05        0.03        inf        True         1.5      1,309           799.4       0.2138      1,900           1,417        0.234
       10ª   1% / 2% / 3% pela confiança         0.05        0.03       0.05       False           1      1,261           787.8       0.1983      1,826           1,312       0.2301
       11ª 1% / 1,5% / 2% pela confiança         0.05        0.03       0.05        True           1      1,153           786.6       0.2141      1,966           1,399       0.2346
       12ª 1% / 1,5% / 2% pela confiança         0.05        0.03        inf       False         1.5      1,269           752.5       0.2272      1,731           1,237       0.2711
referência   1% / 2% / 3% pela confiança        0.075        0.03       0.05       False           1      1,584           909.3       0.2366      2,433           1,556       0.2299
```

