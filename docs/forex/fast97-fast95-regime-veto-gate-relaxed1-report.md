# Fast97 — Fast95 veto gate relaxado para 1 janela negativa no treino

Resposta objetiva: relaxar o gate ajudou a média, mas ainda não gerou
estratégia.

Fast96 exigia que o gate escolhido nas janelas de treino tivesse `0` janelas
negativas. Fast97 repetiu o mesmo método causal, mas permitiu até `1` janela
negativa no treino.

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

## Configuração

Runner:

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --output-root backend/data/microstructure_v1/reports/fast96-fast95-regime-veto-gate-audit-relaxed1 \
  --maximum-train-negative-labels 1
```

O restante da configuração ficou igual ao Fast96:

- modelo: `hist_gradient_payoff`;
- target / breakeven activation: `3 bps`;
- trailing distance: `8 bps`;
- top fraction: `10%`;
- latência: `100ms`;
- custo principal: `1 bps`;
- custo stress: `2 bps`;
- validação externa: leave-one-window-out;
- seleção do gate: apenas janelas de treino, com inner leave-one-window-out.

## Status

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Motivos:

- `profit_positive_holdouts:6/10`;
- `stress_positive_holdouts:5/10`;
- `profit_negative_holdouts:1`;
- `no_strict_training_survivors:3`;
- `do_not_promote_to_strategy`.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| Holdouts | `10` |
| Training survivors | `7` |
| Sem survivor no treino | `3` |
| Profit-positive @ 1 bps | `6/10` |
| Stress-positive @ 2 bps | `5/10` |
| Profit-negative | `1` |
| Holdout trades avaliados | `53` |
| Holdout net total @ 1 bps | `+134.49723947077848` bps |
| Holdout net total @ 2 bps | `+81.49723947077847` bps |
| Pior holdout @ 1 bps | `-22.056437578171884` bps |

## Por janela

| Holdout | Status | Execuções one-position | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|
| fast16 | stress-positive | `4` | `+5.853442514906057` | `+1.8534425149060567` |
| fast20 | sem survivor | `0` | `0` | `0` |
| fast21 | sem survivor | `0` | `0` | `0` |
| fast22 | negativo | `4` | `-22.056437578171884` | `-26.056437578171884` |
| fast24 | stress-positive | `7` | `+23.126753016852717` | `+16.126753016852717` |
| fast28 | stress-fragile | `8` | `+0.5019148576778534` | `-7.498085142322147` |
| fast29 | stress-positive | `12` | `+22.431389814763165` | `+10.431389814763165` |
| fast32 | stress-positive | `3` | `+4.30090752217628` | `+1.3009075221762796` |
| fast35 | sem survivor | `0` | `0` | `0` |
| fast36 | stress-positive | `15` | `+100.33926932257428` | `+85.33926932257428` |

Observação: este run usou `minimum_holdout_trades=1`, mais permissivo que o
critério operacional anterior de `5` execuções por janela. Se exigirmos 5
execuções, fast16 e fast32 também deixam de contar como janelas robustas.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Relaxar o treino de `0` para `1` janela negativa não resolve o problema:

- ainda há 3 holdouts sem nenhum gate treinável;
- fast22 continua quebrando;
- fast28 só é positivo a `1 bps`, mas falha a `2 bps`;
- parte dos positivos tem amostra pequena demais para operar.

## Conclusão prática

O caminho "top-p + trailing + gate causal em cima do Fast95" ficou fraco:

- Fast96 estrito: `1/10` holdouts positivos;
- Fast97 relaxado: `6/10` positivos a `1 bps`, `5/10` a `2 bps`, com um holdout
  negativo e três sem survivor.

O próximo ramo mais coerente é testar operações maiores/horizonte maior, onde o
custo fixo em bps pesa menos.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast96-fast95-regime-veto-gate-audit-relaxed1/report.json`;
- report sha256:
  `7f7033fc6f5c11c2366798578c5659a378a6ac35225b0e41b1aba4e3c5cf9f12`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast96-fast95-regime-veto-gate-audit-relaxed1/holdout_results.jsonl`;
- holdout results sha256:
  `b3c9739fdcac90b94cb7a8feaeee372e8a77b21e1ffbf3d3ce3b87cf51c94172`.
