# C13-A: stop 1 ATR / alvo 0,5R contra a gestão congelada (teste do Igor, exploratório)

Último ano de dado (2025-09-01 a 2026-09-01). As mesmas candidatas do modelo, geridas pelas duas regras: a congelada (stop 1 ATR, alvo 3R) e a nova (stop 1 ATR, alvo 0,5R — o mesmo stop, um alvo bem mais perto). `media_ganho_R`/`media_perda_R`: a média de R só das operações que ganharam / só das que perderam. Nos dois limiares em uso: 0,05 (a descoberta) e 0,02 (o que está rodando ao vivo).

```
 limiar                                 gestao  trades  win_rate  media_ganho_R  media_perda_R  mean_net  mean_stress     t  total_R  trades_per_day
   0.02       stop 1 ATR · alvo 3R (congelada)   18603    0.5621         0.4891        -0.4041    0.0979      0.07433  5.05    1,821           71.55
   0.02 stop 1 ATR · alvo 0,5R (teste do Igor)   18603    0.6213         0.3814        -0.4186   0.07843      0.05485 6.254    1,459           71.55
   0.05       stop 1 ATR · alvo 3R (congelada)    4061    0.6336         0.5907        -0.4461    0.2108       0.1857 4.901    856.1           15.62
   0.05 stop 1 ATR · alvo 0,5R (teste do Igor)    4061    0.7225          0.421        -0.4957    0.1666       0.1415  6.31    676.5           15.62
```
