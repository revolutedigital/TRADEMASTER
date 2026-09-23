# Fast91 — leave-one-label-out gate audit

Resposta objetiva: não encontramos um gate causal simples que generalize. O
melhor gate escolhido em treino quase sempre usa `directed_flow_imbalance_5s`
alto, mas quando cada janela é deixada fora do treino, ele só fica positivo em
`5/12` holdouts.

Fast91 é diagnóstico pós-outcome, não estratégia.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Pergunta testada

Fast90 mostrou que baixar o top-p não cria edge. A próxima pergunta era:

> Dá para selecionar, só com labels de treino, um gate causal simples que
> remova as janelas ruins e preserve as boas?

Fast91 usa leave-one-label-out:

1. deixa uma label/WAL fora;
2. escolhe o melhor base+gate usando todas as outras labels;
3. aplica o gate congelado na label deixada fora;
4. mede o resultado one-position em stress bps.

Isso força o filtro a provar que generaliza entre janelas.

## Validação do runner

Teste unitário:

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast91_lolo_gate_audit.py -q --no-cov
```

Resultado: `8 passed`.

Lint:

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast91_lolo_gate_audit.py \
  backend/tests/unit/test_fast91_lolo_gate_audit.py
```

Resultado: `All checks passed!`.

## Comando

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast91_lolo_gate_audit.py \
  --input-path backend/data/microstructure_v1/reports/fast90-near-threshold-scout-all-local/independent_settlements.jsonl \
  --output-root backend/data/microstructure_v1/reports/fast91-lolo-gate-audit \
  --probability-margins 0.02,0.05
```

## Resultado

- status: `FAST91_LOLO_GATE_FAILED`;
- input rows: `2.517`;
- holdouts: `12`;
- holdouts positivos: `5`;
- holdouts negativos: `5`;
- holdouts sem trades suficientes: `2`;
- trades em holdout: `24`;
- stress total em holdout: `+6.430233183860359` bps;
- stress médio por label: `+0.5358527653216966` bps;
- pior holdout: `-23.8146319649807` bps.

O agregado ficou levemente positivo porque algumas janelas compensaram outras,
mas isso não é robustez: metade das labels úteis falhou.

## Holdouts

| Holdout | Status | Base treinado | Gate treinado | Trades | Stress |
|---|---|---|---|---:|---:|
| fast16 | insuficiente | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6171871857` | 0 | `0.0` |
| fast20 | negativo | `near020_all_veto` | `directed_flow_imbalance_5s >= 0.5682731125` | 2 | `-23.8146319649807` |
| fast21 | negativo | `near020_return5_and_flow5_positive_top_probability` | `realized_vol_5s_bps <= 0.3074777876` | 3 | `-2.9208653629184873` |
| fast22 | positivo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6145813931` | 2 | `+7.155760776965277` |
| fast24 | positivo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6171871857` | 3 | `+1.6917637396274965` |
| fast28 | negativo | `near020_return5_and_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6534035568` | 2 | `-10.548249858774327` |
| fast29 | negativo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6109854445` | 1 | `-3.099596542761195` |
| fast32 | insuficiente | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6171871857` | 0 | `0.0` |
| fast35 | positivo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6169008116` | 5 | `+26.58667992500286` |
| fast36 | negativo | `near020_all_veto` | `probability >= 0.8921458205` | 3 | `-1.7668049430615973` |
| fast82 | positivo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6171871857` | 2 | `+8.529050178333772` |
| fast83 | positivo | `near020_flow5_positive_top_probability` | `directed_flow_imbalance_5s >= 0.6169008116` | 1 | `+4.617127236427259` |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast91 elimina uma leitura otimista do Fast87/Fast90:

- o `directed_flow_imbalance_5s` alto é realmente o filtro que o treino tenta
  escolher;
- mas ele não segura quando aplicado em labels fora do treino;
- a estratégia ainda depende de quais janelas aparecem no futuro;
- isso é frágil demais para bot autônomo.

## Implicação

A família `top-p / near-threshold + flow5` está quase esgotada como caminho de
curto prazo. O próximo passo precisa ser uma destas opções:

1. continuar coletando janelas prospectivas para Fast88/Fast91, aceitando que a
   frequência é baixa;
2. construir uma família de entrada diferente, sem depender do top-p atual;
3. mudar o problema para operações menos curtas, onde custo e ruído pesam menos.

O que não parece mais racional: baixar threshold e tentar filtrar com um único
gate de flow/volatilidade.

## Arquivos e hashes

- Fast91 runner:
  `5ac45474475b483303a1bb1d1a20f2c73c1555184312d31a13f33862b7ca503c`;
- Fast91 tests:
  `3d53e41f826f890e3ac37b6ced0723b00a200c06d2ada5a2aefd976a795c51d2`;
- Fast91 report:
  `a9557c82badf4fde7890d8edc1de09815287dc5210292864fe4b6d810f8e965f`;
- Fast91 holdout results:
  `bbe00f6216410c19ba8696df05a82e77ef7c8e30726e94586e7c3913aba83334`.
