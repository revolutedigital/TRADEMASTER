# C15: prêmio do dia de FOMC nas moedas (H1) e S&P 500 no pico do VIX (H2)

Commit do código: 960d88068f401420bb038a3bf25c981992156857. Pré-registro: `docs/forex/c15-preregistration.md` (sha256 51861325aaec8e64fbf4a86a52d85e8cc175efed0fab4b5e345e41a0a4136e98).

**H1 aprovada: não. H2 aprovada: não.** Holm sobre as duas.

## H1 — dia de FOMC (principal: 2014-01 a 2026-08; o resto é informativo)

```
{
 "main": {
  "events": 75,
  "mean_net_bp": 11.48073700326616,
  "mean_stress_bp": 9.664109430740076,
  "mean_net_r": 0.12379709978831499,
  "t": 1.4275415968531477,
  "p_one_sided": 0.07671192206951208,
  "other_days_bp": -2.583748635842226,
  "other_days": 913,
  "positive_years": 9,
  "years": 12,
  "legs_per_event": 2.48,
  "placebo_means_bp": [
   -5.045805279030117,
   9.048316787233093,
   3.575515337731504,
   2.1567067496484773,
   0.4149407336211564,
   -4.12555374458947,
   9.891141908241647,
   3.1301694877100483,
   -4.301522283821333,
   1.3932630491491715,
   -6.0488863715804,
   1.6350209117430423,
   6.280851989340685,
   -0.24818989557393364,
   7.696747174351623,
   -2.0926061419932576,
   -2.9609870208974436,
   -7.058458478938759,
   -1.2167400209482326,
   -4.240529997256345
  ],
  "placebo_reaching": 0,
  "approved_before_holm": false,
  "p_holm": 0.15342384413902416,
  "approved": false
 },
 "informativo 2003-2011": {
  "events": 71,
  "mean_net_bp": 20.484123954994246,
  "mean_stress_bp": 18.88916588241178,
  "mean_net_r": 0.11241877736384527,
  "t": 1.9370700638937441,
  "p_one_sided": 0.026368389698162728,
  "other_days_bp": 1.0301026303387015,
  "other_days": 1314,
  "positive_years": 7,
  "years": 9,
  "legs_per_event": 3.915492957746479
 },
 "informativo 2012-2013": {
  "events": 16,
  "mean_net_bp": -0.636594934900919,
  "mean_stress_bp": -2.077211117296554,
  "mean_net_r": 0.0460032089961171,
  "t": -0.045228666363340456,
  "p_one_sided": 0.5180374774146683,
  "other_days_bp": -2.4991252968184487,
  "other_days": 293,
  "positive_years": 1,
  "years": 2,
  "legs_per_event": 5.625
 }
}
```

## H2 — S&P 500 no pico do VIX (2003-01 a 2026-08)

```
{
 "trades": 102,
 "mean_net_bp": 40.45980687914143,
 "mean_stress_bp": 27.606462801784474,
 "t": 0.8760875689261591,
 "p_one_sided": 0.190491215086102,
 "baseline_bp": 15.896215877839369,
 "positive_years": 15,
 "years": 24,
 "vix_sha256": "4bca3f49062d5196f00cf6e62d04f77304608884ca4d791d2e7210e274d52796",
 "approved_before_holm": false,
 "p_holm": 0.190491215086102,
 "approved": false
}
```
