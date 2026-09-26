# C13, frente A: placebo antes do período intocado (emenda 2)

Commit do código: 0dcf055aa3df94f1a3e71e1241fd429edc9aa303. Os minutos da descoberta com o sinal de cada movimento trocado ao acaso (semente 20261026), e a mesma banca (profundidade 3, peso 2000, τ = 0.05) com o mesmo walk-forward. Sem vantagem possível, a média aceita não pode ser positiva com t ≥ 2. **Limpo: sim.**

```
threshold  trades  mean_net  mean_stress  win_rate       t  p_one_sided  market_share  markets  trades_per_day
        0  168906 -0.008933     -0.02131    0.4854  -2.415       0.9921        0.3243       37           60.89
     0.05    3026  -0.02718     -0.03743     0.462 -0.8604       0.8052        0.3939       33           1.091
    todas 2421423  -0.01987     -0.04081    0.4783  -37.96            1             0       37           872.9
```
