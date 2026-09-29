# C13-A: stops e alvo mais apertados contra a gestão congelada (teste do Igor, exploratório)

Último ano de dado (2025-09-01 a 2026-09-01) — leitura única, não é a disciplina de escolher em 2016-2022 e ler em 2023-2026 que a gestão congelada passou; um número bom aqui não é aval pra trocar, é candidata pra passar por esse teste honesto antes de ir pro robô. As mesmas candidatas do modelo, geridas por 6 regras: a congelada (stop 1 ATR, alvo 3R) e 5 combinações de stop mais apertado (1 / 0,5 / 0,25 ATR) com o alvo de sempre ou o alvo mais perto (0,5R). `media_ganho_R`/`media_perda_R`: a média de R só das operações que ganharam / só das que perderam. Nos dois limiares em uso: 0,05 (a descoberta) e 0,02 (o que está rodando ao vivo).

```
 limiar                                            gestao  trades  win_rate  media_ganho_R  media_perda_R  mean_net  mean_stress     t  total_R  trades_per_day
   0.02                  stop 1 ATR · alvo 3R (congelada)   18603    0.5621         0.4891        -0.4041    0.0979      0.07433  5.05    1,821           71.55
   0.02                            stop 1 ATR · alvo 0,5R   18603    0.6213         0.3814        -0.4186   0.07843      0.05485 6.254    1,459           71.55
   0.02                            stop 0,5 ATR · alvo 3R   18603    0.5286         0.9576        -0.7413    0.1567       0.1095  4.92    2,915           71.55
   0.02      stop 0,5 ATR · alvo 0,5R (os dois apertados)   18603    0.7255          0.434        -0.7949   0.09667      0.04953 6.808    1,798           71.55
   0.02                           stop 0,25 ATR · alvo 3R   18603    0.4303          1.881         -1.016    0.2303        0.136 5.104    4,284           71.55
   0.02 stop 0,25 ATR · alvo 0,5R (os dois bem apertados)   18603    0.7597         0.4017         -1.047   0.05349      -0.0408  3.69      995           71.55
   0.05                  stop 1 ATR · alvo 3R (congelada)    4061    0.6336         0.5907        -0.4461    0.2108       0.1857 4.901    856.1           15.62
   0.05                            stop 1 ATR · alvo 0,5R    4061    0.7225          0.421        -0.4957    0.1666       0.1415  6.31    676.5           15.62
   0.05                            stop 0,5 ATR · alvo 3R    4061    0.5902           1.13        -0.7715    0.3507       0.3005 5.017    1,424           15.62
   0.05      stop 0,5 ATR · alvo 0,5R (os dois apertados)    4061    0.8197         0.4409        -0.8892    0.2011       0.1509 7.182    816.8           15.62
   0.05                           stop 0,25 ATR · alvo 3R    4061    0.4883          2.174          -1.04    0.5291       0.4287 4.719    2,149           15.62
   0.05 stop 0,25 ATR · alvo 0,5R (os dois bem apertados)    4061     0.806         0.3958         -1.076    0.1102     0.009741 4.082    447.4           15.62
```
