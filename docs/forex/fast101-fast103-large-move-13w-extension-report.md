# Fast101–Fast103 — extensão do candidato maior para 13 janelas

Resposta objetiva: adicionar fast82, fast83 e fast89 confirmou que existe sinal
no candidato maior, mas o gate ainda não transforma isso em estratégia.

Fast98–Fast100 mostraram que operações maiores (`target 10 bps`, trailing
distance `15 bps`, top `5%`) eram o ramo mais promissor, porém raro. Fast101
estendeu esse candidato das 10 janelas anteriores para 13 janelas, adicionando:

- fast82;
- fast83;
- fast89.

Depois Fast102/Fast103 testaram gate causal nesse conjunto ampliado.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Fast101 — payoff scout focado em 13 janelas

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_payoff_model_scout.py \
  --wal-roots fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m,fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m,fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m,fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m \
  --output-root backend/data/microstructure_v1/reports/book-payoff-model-scout-fast101-large-move-13w-target10-trail15-top5-300s-100ms \
  --horizon-seconds 300 \
  --target-bps-values 10 \
  --trailing-distance-bps-values 15 \
  --top-fractions 0.05 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 11 \
  --minimum-executable-per-window 3
```

Status:

`PAYOFF_MODEL_FAILED_REPLICATION`

Resumo do melhor candidato (`hist_gradient_payoff`, target `10`, trailing `15`,
top `5%`):

| Métrica | Valor |
|---|---:|
| Holdouts | `13` |
| One-position execuções | `42` |
| One-position net @ 1 bps | `+13.763261447660312` bps |
| One-position net @ 2 bps | `-28.236738552339688` bps |
| Positive windows @ 1 bps | `11/13` |
| Positive windows @ 2 bps | `11/13` |
| Janelas com mínimo de 3 execuções | `9/13` |
| Pior janela @ 1 bps | `-14.226994710722115` bps |

As novas janelas foram boas:

| Janela nova | Execuções | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|
| fast82 | `4` | `+6.8064677378686` | `+2.8064677378685996` |
| fast83 | `4` | `+15.978204811527192` | `+11.978204811527192` |
| fast89 | `3` | `+8.96503860906158` | `+5.96503860906158` |

Mas dois outliers quebraram a estratégia:

- fast24: `-14.016812888603907` bps @ 1 bps;
- fast36: `-113.81595768577692` bps @ 1 bps.

## Fast102 — gate causal estrito em 13 janelas

Comando principal:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --wal-roots [13 janelas] \
  --output-root backend/data/microstructure_v1/reports/fast102-large-move-13w-regime-veto-gate-audit \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.05 \
  --minimum-train-trades 20 \
  --minimum-train-labels 10 \
  --minimum-holdout-trades 1 \
  --maximum-train-negative-labels 0
```

Status:

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Resumo:

| Métrica | Valor |
|---|---:|
| Holdouts | `13` |
| Training survivors | `3` |
| Sem survivor no treino | `10` |
| Profit-positive @ 1 bps | `1/13` |
| Stress-positive @ 2 bps | `1/13` |
| Profit-negative | `2` |
| Holdout trades avaliados | `11` |
| Holdout net @ 1 bps | `-30.911266014541148` bps |
| Holdout net @ 2 bps | `-41.91126601454115` bps |

O gate estrito ficou rígido demais e não encontrou survivor em `10/13`
holdouts.

## Fast103 — gate causal relaxado para 1 janela negativa

Mesma configuração do Fast102, mas com:

```bash
--maximum-train-negative-labels 1
```

Status:

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Resumo:

| Métrica | Valor |
|---|---:|
| Holdouts | `13` |
| Training survivors | `8` |
| Sem survivor no treino | `5` |
| Profit-positive @ 1 bps | `5/13` |
| Stress-positive @ 2 bps | `5/13` |
| Profit-negative | `3` |
| Holdout trades avaliados | `24` |
| Holdout net @ 1 bps | `-6.588620482745904` bps |
| Holdout net @ 2 bps | `-30.588620482745903` bps |

Por janela relevante:

| Holdout | Status | Execuções | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|
| fast16 | positivo | `1` | `+4.343264375832244` | `+3.343264375832244` |
| fast22 | positivo | `1` | `+3.625392107392323` | `+2.625392107392323` |
| fast24 | negativo | `3` | `-23.115327749933726` | `-26.115327749933726` |
| fast35 | positivo | `4` | `+10.92291370261829` | `+6.92291370261829` |
| fast36 | negativo | `5` | `-34.88933011229763` | `-39.88933011229763` |
| fast82 | positivo | `4` | `+22.357780694561953` | `+18.357780694561953` |
| fast83 | positivo | `4` | `+11.134615192370765` | `+7.134615192370765` |
| fast89 | negativo | `2` | `-0.9679286932901245` | `-2.9679286932901245` |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O achado importante é positivo, mas insuficiente:

- a extensão para fast82/83/89 não destruiu o sinal bruto;
- o candidato maior ficou positivo em `11/13` janelas no one-position;
- porém fast24/fast36 carregam prejuízos grandes;
- o gate causal, tanto estrito quanto relaxado, não aprendeu a filtrar esses
  regimes sem destruir a maioria dos holdouts.

## Conclusão prática

O ramo de operações maiores é o melhor até agora, mas ainda não é estratégia.

Se for continuar nele, o próximo passo real não é mexer em top-p manualmente na
mesma base. O próximo passo é um destes:

1. coletar/baixar mais janelas de livro para aumentar a amostra;
2. criar uma arquitetura de regime específica para detectar outlier de fast36,
   mas pré-registrando antes;
3. testar outro mercado/par com menor microstructure noise.

## Arquivos e hashes

- Fast101 report:
  `14e957641195fe6526f2b01f1a69e400d891123aedda2642012f40226d12989f`;
- Fast102 report:
  `1239c65a5d32df02c7ff0d1528790c4cb4223df45d0992c5e2ba54e115af6f5d`;
- Fast102 holdouts:
  `5b5e59d21b3873e25590e348f7dd0e27516948feef1184fa9b1f7dc16991710d`;
- Fast103 report:
  `418633aa4aebeb5e2e28990714f413d99f4e568bba1a53ad8a17db7ba1eb0785`;
- Fast103 holdouts:
  `af64b3303f19fbed96e66651fe417028090347013c7e4976c89b924c407143c0`.
