# Fast93 — trailing payoff gate audit

Resposta objetiva: ainda não temos estratégia.

Fast92 matou a hipótese de "abrir mais entradas". Fast93 testou a próxima
pergunta certa para o bot real descrito: existe algum gate causal pré-trade que
já paga o trailing/stop no treino e sobrevive quando escondemos um WAL label?

Resultado: não. Nem o treino encontrou survivor limpo.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Fonte

Entrada:

`backend/data/microstructure_v1/reports/fast92-broad-universe-scout-all-local/independent_settlements.jsonl`

Linhas: `6.579` settlements completos.

Labels: `12`.

O Fast93 usa apenas features causais pré-trade:

- `probability`;
- `threshold_distance`;
- `realized_vol_30s_bps`;
- `realized_vol_5s_bps`;
- `quote_volume_30s`;
- `trade_count_30s`;
- `mean_interarrival_ms_30s`;
- `directed_flow_imbalance_1s/5s/30s`;
- `directed_return_1s/5s/30s`.

Não usa `gross_bps`, MFE/MAE, `exit_reason` ou qualquer informação futura como
feature de seleção.

## Método

Para cada label escondido:

1. treina em todos os outros labels;
2. testa policies de conflito `top_probability` e `veto`;
3. testa gates `gte/lte` em quantis `0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9`;
4. aplica one-position constraint;
5. só aceita candidato de treino se:
   - tiver pelo menos `10` trades;
   - cobrir pelo menos `3` labels;
   - stress total for positivo;
   - stress médio for positivo;
   - labels negativos ficarem dentro do limite configurado;
6. aplica o gate congelado no label escondido.

Isso mede diretamente o que importa: payoff líquido do trailing sob custo de
stress, não probabilidade de entrada.

## Resultado strict

Com limite de `0` labels negativos no treino:

- status: `FAST93_NO_STRICT_TRAINING_SURVIVORS`;
- holdouts: `12`;
- training survivors: `0/12`;
- holdout trades: `0`;
- positive holdouts: `0/12`.

Interpretação: usando o critério limpo, não existe nem candidato treinável.

## Resultado relaxado 1

Com limite de `1` label negativo no treino:

- status: `FAST93_NO_STRICT_TRAINING_SURVIVORS`;
- holdouts: `12`;
- training survivors: `0/12`;
- holdout trades: `0`;
- positive holdouts: `0/12`.

Interpretação: a falha não é excesso de rigidez no critério zero-negative.
Mesmo permitindo um label ruim, nenhum gate causal paga o trailing no treino.

## Resultado relaxado 5

Com limite de `5` labels negativos no treino:

- status: `FAST93_TRAILING_PAYOFF_GATE_FAILED`;
- holdouts: `12`;
- no-survivor holdouts: `11`;
- holdout trades: `19`;
- positive holdouts: `0/12`;
- negative holdouts: `1`;
- holdout stress total: `-86.00211136779198` bps.

O único candidato selecionado no modo muito relaxado:

- policy: `top_probability`;
- gate: `mean_interarrival_ms_30s <= 12.023886639676114`;
- treino: `24` trades, `9` labels, stress `+15.426479489766358` bps;
- treino já tinha `5` labels negativos;
- holdout `prospective-wal-fast36-30m`: `19` trades, stress
  `-86.00211136779198` bps.

Interpretação: quando afrouxamos até aceitar treino sujo, o primeiro sinal que
aparece quebra feio fora da amostra.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast93 elimina a hipótese:

> "talvez um gate causal simples consiga escolher entradas que pagam o trailing
> no universo amplo Fast92."

Não consegue.

## O que isso diz sobre o erro atual

Estamos tentando tirar edge de um setup em que:

1. a média bruta é pequena;
2. o custo transforma média levemente positiva em negativa;
3. o trailing quase nunca captura movimento grande;
4. os filtros que parecem bons não são estáveis por label;
5. quando exigimos seleção honesta no treino, não sobra candidato.

O problema não é só calibrar top-p. O problema é que o feature set atual não
identifica com estabilidade os momentos em que o movimento futuro paga custo,
stop e trailing.

## Próxima direção

O próximo teste não deve insistir em gates simples sobre o Fast92. Caminhos com
maior chance:

1. reconstruir o label para "payoff líquido mínimo com cauda ruim penalizada";
2. separar explicitamente regimes de volatilidade/liquidez antes de treinar;
3. aumentar horizonte/payoff mínimo para buscar menos trades e margem maior;
4. capturar features de livro completas no shadow, porque WAL recente ainda é
   pobre para o modelo non-stop completo.

## Validação

Comandos:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/unit/test_fast93_trailing_payoff_gate_audit.py \
  -q --no-cov
```

Resultado: `7 passed`.

```bash
./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast93_trailing_payoff_gate_audit.py \
  backend/tests/unit/test_fast93_trailing_payoff_gate_audit.py
```

Resultado: `All checks passed!`.

## Arquivos e hashes

- script:
  `backend/scripts/research/run_fast93_trailing_payoff_gate_audit.py`;
- teste:
  `backend/tests/unit/test_fast93_trailing_payoff_gate_audit.py`;
- strict report:
  `557c6edf2c7672f7c27a7c86f395341d243511c43ce2499a5e21ded5ef8c37c6`;
- strict holdouts:
  `872488089bef5a5ce5677c670b5725404013862fd9a400ba73862f6ec7621fdb`;
- relaxed1 report:
  `a0333d32c7eb4a980552355bccb5c013206a30052f629efe1e922e27088aa729`;
- relaxed1 holdouts:
  `872488089bef5a5ce5677c670b5725404013862fd9a400ba73862f6ec7621fdb`;
- relaxed5 report:
  `9850d0d93f1db5dfe954490bc5a0644d266ae3a007bfb530162866b868f4e4a1`;
- relaxed5 holdouts:
  `907432de4aa2d5f23c9eed46ddf9a97a275eec28bce45ab6affd9295810605e0`.
