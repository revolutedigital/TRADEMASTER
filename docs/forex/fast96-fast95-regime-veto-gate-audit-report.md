# Fast96 — veto gate causal para o candidato Fast95

Resposta objetiva: ainda não temos estratégia.

Fast96 testou a continuação mais promissora do Fast95: pegar o candidato
near-edge (`hist_gradient_payoff`, target `3 bps`, trailing distance `8 bps`,
top `10%`) e tentar adicionar um gate causal de no-trade/regime.

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

## Pergunta testada

Fast95 ficou positivo em `9/10` janelas no modo one-position a `1 bps`, mas
falhou em replicação estrita. Fast96 testou:

> Depois que o modelo escolhe o top-p, existe um gate causal que veta os regimes
> ruins sem olhar o holdout?

O gate foi aplicado como veto depois do top-p. Isso é mais conservador que
"gate antes e preenche com trades piores": o gate só reduz trades.

## Configuração

Runner novo:

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py
```

Configuração efetiva:

- janelas: Fast16, Fast20, Fast21, Fast22, Fast24, Fast28, Fast29, Fast32,
  Fast35, Fast36;
- modelo: `hist_gradient_payoff`;
- horizonte: `120s`;
- target / breakeven activation: `3 bps`;
- trailing distance: `8 bps`;
- top fraction: `10%`;
- gate keep fractions: `0.50`, `0.70`, `0.85`, `1.00`;
- latência: `100ms`;
- custo principal: `1 bps`;
- custo stress: `2 bps`;
- validação externa: leave-one-window-out;
- seleção do gate: apenas janelas de treino, com inner leave-one-window-out.

## Status

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Motivos principais:

- `profit_positive_holdouts:1/10`;
- `stress_positive_holdouts:1/10`;
- `no_strict_training_survivors:9`;
- `do_not_promote_to_strategy`.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| Holdouts | `10` |
| Training survivors | `1` |
| No strict training survivor | `9` |
| Profit-positive holdouts @ 1 bps | `1/10` |
| Stress-positive holdouts @ 2 bps | `1/10` |
| Holdout trades avaliados | `12` |
| Holdout net total @ 1 bps | `+22.431389814763165` bps |
| Holdout net total @ 2 bps | `+10.431389814763165` bps |

O único holdout com gate treinável e positivo foi `fast29`.

## Único bolso sobrevivente

Holdout: `fast29`

Gate selecionado:

`directed_spot_perp_return_gap_1s_bps >= 0`

Resumo do holdout fast29:

- one-position executions: `12`;
- net total @ `1 bps`: `+22.431389814763165` bps;
- net total @ `2 bps`: `+10.431389814763165` bps.

Isso não é estratégia. É um bolso específico. O mesmo método não encontrou gate
treinável para `9/10` holdouts.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast96 respondeu a pergunta principal: o caminho "top-p + trailing + veto gate"
não ficou robusto com as janelas atuais. O Fast95 segue sendo um near-edge
interessante, mas não operacional.

## O que isso significa

O problema não é que trailing stop "não funciona". O problema é que a entrada
continua instável:

- quando a entrada está no regime certo, o trailing carrega bem;
- quando a entrada cai no regime errado, o trailing só limita ou posterga a
perda;
- o modelo ainda não identifica causalmente, de forma replicável, quando está
no regime certo.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast96-fast95-regime-veto-gate-audit/report.json`;
- report sha256:
  `60d0a5a8cbc7f9ee8f295799949396461a1c52e04fddb9a83c8ad1b7fc681f96`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast96-fast95-regime-veto-gate-audit/holdout_results.jsonl`;
- holdout results sha256:
  `c6efa10ca96a95d663069162b7f30932faa33eea2003442b1c0cef278b636808`.

## Verificação

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast96_fast95_regime_gate_audit.py -q --no-cov
```

Resultado: `6 passed`.

```bash
./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  backend/tests/unit/test_fast96_fast95_regime_gate_audit.py
```

Resultado: `All checks passed!`.
