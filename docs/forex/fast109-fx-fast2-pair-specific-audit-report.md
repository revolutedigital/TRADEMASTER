# Fast109 — auditoria por par do grid FX Fast2

Resposta objetiva: ainda não temos estratégia.

Fast2 rejeitou as 212 combinações no agregado. Fast109 testou se essa rejeição
estava escondendo um edge em algum par específico. O resultado encontrou alguns
bolsões fracos positivos, mas nenhum passou estabilidade anual strict.

## Safety boundary

Mantido em todos os passos:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Escopo auditado

Fonte:

- mesmo maquinário de simulação do Fast2;
- amostra `discovery` do Fast2 (`S0`, 2014-11 a 2018-11);
- matrizes M1 em `backend/data/lab_pre`;
- grid em `docs/forex/fast2-grid.json`.

Escopo:

| Métrica | Valor |
|---|---:|
| Configurações Fast2 | `212` |
| Avaliações config+par | `1580` |
| Pares/janelas avaliados | `11` pares, incluindo `AUDNZD` sintético |
| Status | `FAST109_FX_FAST2_PAIR_SPECIFIC_WEAK_ONLY_FAILED_STABILITY` |
| Estratégia aprovada | `false` |

## Critério usado no diagnóstico

Critério weak:

- pelo menos `100` trades em base e stress;
- pelo menos `12` meses;
- pelo menos `3` anos;
- média R positiva em base;
- média R positiva em stress.

Critério strict:

- todos os critérios weak;
- todos os anos positivos em base;
- todos os anos positivos em stress.

Esse é um diagnóstico pós-hoc por par. Mesmo que houvesse candidato strict, ele
não poderia virar execução direto; precisaria de novo pré-registro e validação
prospectiva.

## Resultado

| Métrica | Valor |
|---|---:|
| Weak candidates | `25` |
| Strict candidates | `0` |
| Maior média stress R | `+0.04749825724552229` |
| Maior stress t | `0.759242032673688` |

O melhor caso foi:

| Config | Par | Trades | Média base R | Média stress R | Anos stress positivos | Pior ano stress | Stress t |
|---|---|---:|---:|---:|---:|---:|---:|
| `F1a.rr2.5.w0.3-0.8.se09:30.x16:30` | `GBPUSD` | `242` | `+0.09401292570440892` | `+0.04749825724552229` | `3/5` | `-0.0715181680664398` | `0.759242032673688` |
| `F1a.rr2.5.w0.3-1.2.se09:30.x16:30` | `GBPUSD` | `246` | `+0.09194706401250825` | `+0.04578242549460353` | `3/5` | `-0.05678039063922319` | `0.7424664490072271` |
| `F1a.rr1.5.w0.3-0.8.se09:30.x16:30` | `EURJPY` | `367` | `+0.06514023939849498` | `+0.028301673134730164` | `2/5` | `-0.04080607080022978` | `0.6890357279820257` |
| `F1b.rr1.5.w0.6-2.0.se09:30.x16:00` | `EURUSD` | `129` | `+0.049665` | `+0.025151` | `3/5` | `-0.136932` | `0.521` |

Distribuição dos weak candidates:

| Corte | Distribuição |
|---|---|
| Família | `F1a: 16`, `F1b: 9` |
| Par | `GBPUSD: 11`, `EURUSD: 6`, `EURJPY: 5`, `USDJPY: 3` |

Isso aponta para uma pista de breakout de sessão, não para uma estratégia pronta.
O sinal positivo é pequeno, instável e estatisticamente fraco.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast109 responde a pergunta “será que o Fast2 falhou só porque misturou pares
ruins com pares bons?”: não o suficiente. Há pares/configurações menos ruins,
mas nenhum candidato manteve stress positivo em todos os anos.

## O que isso ensina

- Existe uma pista recorrente em breakout de sessão, principalmente `GBPUSD`.
- O edge bruto/stress positivo aparece em alguns recortes, mas some em anos
  específicos.
- O custo ainda deixa a margem pequena demais.
- Trailing/top-p não resolvem se a entrada só funciona em alguns regimes.
- O próximo teste honesto não é ligar bot; é pré-registrar uma hipótese estreita
  de breakout por par/regime e observar prospectivamente.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast109_fx_fast2_pair_specific_audit.py \
  --stage discovery --progress
```

## Verificações

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast109_fx_fast2_pair_specific_audit.py -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast109_fx_fast2_pair_specific_audit.py \
  backend/tests/unit/test_fast109_fx_fast2_pair_specific_audit.py
```

Resultado:

- `3 passed`;
- `All checks passed!`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast109/fx-fast2-pair-specific-audit/report.json`;
- report sha256:
  `b5e8a2412c33eafebfdd6a4dbd90b88a28ccdd4d043ed1bf6d21e46c06c231f0`;
- weak candidates sha256:
  `1cb27920d2a529feb6c27e199e95cb107daa68752f946e564d115095590805fd`;
- strict candidates sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- top diagnostics sha256:
  `e68e249157bcfae40abda85014de22bc3f8b0732377229470a40ba2ef108b9f2`.
