# Fast110 — validação out-of-sample dos weak candidates do Fast109

Resposta objetiva: ainda não temos estratégia aprovada, mas encontramos a pista
mais limpa até aqui.

Fast109 achou 25 bolsões weak dentro do Fast2 por par. Fast110 congelou essa
lista e validou somente esses pares/configurações nas amostras seguintes do
Fast2:

- `replication` / S1: 2019-01 a 2024-08;
- `confirmation` / S2: 2024-09 a 2026-08.

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

## Escopo validado

Fonte:

- weak candidates congelados do Fast109:
  `backend/data/lab_fast109/fx-fast2-pair-specific-audit/weak_pair_config_candidates.json`;
- source sha256:
  `1cb27920d2a529feb6c27e199e95cb107daa68752f946e564d115095590805fd`;
- mesmo maquinário de simulação do Fast2/Fast109.

Escopo:

| Métrica | Valor |
|---|---:|
| Candidatos congelados | `25` |
| Configurações únicas | `18` |
| Pares únicos | `4` |
| Strict survivors | `0` |
| Weak survivors | `3` |
| Estratégia aprovada | `false` |

Status:

`FAST110_FX_FAST2_PAIR_VALIDATION_WEAK_ONLY_FAILED_STABILITY`

## Critério usado

Por estágio:

- `replication`: mínimo `100` trades;
- `confirmation`: mínimo `30` trades;
- média R positiva em base;
- média R positiva em stress.

Para strict:

- passar o critério acima;
- base positiva em todos os anos;
- stress positivo em todos os anos.

Resultado:

- `replication_strict_pass_count = 0`;
- `confirmation_strict_pass_count = 0`;
- nenhum candidato passou strict em S1+S2.

## Weak survivors

Os 3 sobreviventes weak são todos a mesma ideia:

```text
USDJPY
F1a / London breakout
range width 0.6-2.0 ATR
signal_end 11:00
exit 16:30
reward_risk variando 1.0, 1.5, 2.5
```

| Config | S1 trades | S1 stress R | S1 stress years | S1 stress t | S2 trades | S2 stress R | S2 stress years | S2 stress t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `F1a.rr1.0.w0.6-2.0.se11:00.x16:30` | `152` | `+0.010234816076974384` | `4/6` | `0.23400524368225553` | `78` | `+0.0021910801239872083` | `2/3` | `0.03566181697493101` |
| `F1a.rr1.5.w0.6-2.0.se11:00.x16:30` | `152` | `+0.005472684645253482` | `4/6` | `0.12296359960742644` | `78` | `+0.011299489464689203` | `2/3` | `0.1761395800158334` |
| `F1a.rr2.5.w0.6-2.0.se11:00.x16:30` | `152` | `+0.009628735752547975` | `4/6` | `0.20997476462892162` | `78` | `+0.025061390385669378` | `2/3` | `0.36304329155980314` |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Mas esta pista merece tratamento diferente das anteriores: ela foi descoberta em
S0 e continuou positiva em média em S1 e S2, sempre no mesmo par e mesma família
de regra. O problema é que a vantagem é pequena, o `t` é baixo e há anos
negativos. Isso é uma hipótese de regime, não uma máquina pronta.

## O que isso ensina

- A pista mais promissora agora é `USDJPY` + London breakout.
- Não é um edge robusto o bastante para execução.
- O ajuste de reward-risk não é o centro; as 3 versões sobrevivem/fracas.
- O próximo teste precisa perguntar: em que regime esse breakout funciona?
- Se o regime não puder ser definido antes e validado fora da amostra, isso morre
  como overfit fraco.

## Próximo caminho recomendado

Fast111 deve ser um pré-registro/scout disciplinado de regime para essa linha:

- universo travado: somente `USDJPY`;
- família travada: `F1a.w0.6-2.0.se11:00.x16:30`;
- testar filtros simples e explicáveis, como volatilidade diária, range pré-London,
  dia da semana, sessão anterior e direção de tendência;
- separar treino/holdout temporal;
- continuar `research_only=true`.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast110_fx_fast2_pair_validation.py \
  --progress
```

## Verificações

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast109_fx_fast2_pair_specific_audit.py \
  backend/tests/unit/test_fast110_fx_fast2_pair_validation.py -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast109_fx_fast2_pair_specific_audit.py \
  backend/scripts/research/run_fast110_fx_fast2_pair_validation.py \
  backend/tests/unit/test_fast109_fx_fast2_pair_specific_audit.py \
  backend/tests/unit/test_fast110_fx_fast2_pair_validation.py
```

Resultado:

- `7 passed`;
- `All checks passed!`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast110/fx-fast2-pair-validation/report.json`;
- report sha256:
  `dde0f2ef00b580ba685825666f5908c15f8a1af036885422445d9925641d8304`;
- validation results sha256:
  `216ac8abbf1ce81ae6b0b1283311872007873de5e5d98891aed53a27c4ae59e1`;
- strict survivors sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- weak survivors sha256:
  `efccc47f8f5d453d870540c8bb6aa514eba44044ee3e136a2b3d805196578fb7`;
- replication evaluations sha256:
  `c8f7eb58dacbede6cf625bf64e885a7b04bd435278ea4fa550acaea481aa5375`;
- confirmation evaluations sha256:
  `4d3d7062dd9fb92e1526d90517ee0d4a79df6ac97d1382efb115fa83f91d7a02`.
