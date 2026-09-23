# Fast94 — tail-risk utility model audit

Resposta objetiva: ainda não temos estratégia.

Fast93 matou a hipótese de gates causais simples sobre o universo Fast92. Fast94
testou a próxima hipótese mais forte: treinar modelos diretamente no payoff do
trailing, com penalização de cauda ruim, e selecionar sinais por um cutoff
congelado a partir do treino.

Resultado: também falhou leave-one-label-out.

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

Escopo:

- settlements completos: `6.579`;
- labels: `12`;
- candidatos testados: `40`.

## Método

Para cada combinação de modelo e fração top:

1. esconde um WAL label;
2. treina em todos os outros labels;
3. aprende o cutoff de score usando apenas a distribuição do treino;
4. aplica o cutoff congelado no label escondido;
5. aplica one-position constraint;
6. mede stress líquido por holdout.

Isso evita o erro de escolher top-p usando a distribuição do próprio holdout.

Features usadas são causais pré-trade:

- probabilidade e distância para threshold;
- volatilidade realizada;
- quote volume;
- trade count;
- interarrival;
- flow/return direcionais;
- hora;
- lado.

Não usa MFE/MAE, exit reason, `stress_net_bps`, `gross_bps` ou qualquer
informação futura como feature.

## Targets testados

- `stress`: `stress_net_bps`;
- `tail05`: stress com penalização leve de cauda adversa e initial stop;
- `tail10`: stress com penalização mais forte de cauda adversa e initial stop;
- `positive_safe`: classificador para stress positivo sem cauda adversa grande e
  sem initial stop.

Modelos:

- Ridge regression;
- HistGradientBoostingRegressor;
- LogisticRegression;
- HistGradientBoostingClassifier.

Frações:

- `0.5%`;
- `1%`;
- `2%`;
- `5%`;
- `10%`.

## Resultado

Status:

`FAST94_UTILITY_MODEL_FAILED_LOLO`

Motivos principais:

- melhor modelo por ordenação do report: `hgb_tail05`;
- top fraction: `0.1`;
- holdouts positivos: `5/12`;
- holdout stress total: `-127.36220494334309` bps.

Top candidatos:

| Modelo | Target | Top frac | Trades | Holdouts positivos | Holdouts negativos | Vazios | Stress total | Pior holdout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `hgb_tail05` | `tail05` | `0.10` | 47 | 5/12 | 6 | 1 | `-127.36220494334309` | `-104.20818463911638` |
| `hgb_tail10` | `tail10` | `0.05` | 19 | 4/12 | 4 | 4 | `-15.802022828658588` | `-21.40760299368244` |
| `hgb_stress` | `stress` | `0.10` | 40 | 3/12 | 5 | 4 | `-10.113876851756736` | `-21.40760299368244` |
| `hgb_tail05` | `tail05` | `0.05` | 14 | 3/12 | 3 | 6 | `-21.50181345765996` | `-18.002008263499995` |
| `ridge_stress` | `stress` | `0.10` | 66 | 3/12 | 9 | 0 | `-300.4973545446995` | `-130.49532738912842` |

Um candidato (`hgb_stress`, top `0.02`) ficou positivo no agregado, mas só
gerou `1` trade em `1` holdout e deixou `11/12` holdouts vazios. Isso não é
estratégia; é silêncio com um acerto isolado.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast94 elimina a hipótese:

> "talvez um modelo de utilidade/payoff com penalização de cauda consiga salvar
> o universo Fast92."

Não salva. O treino frequentemente acha regiões que parecem boas, mas o score
não transfere de forma estável para os labels escondidos.

## Interpretação

Essa é uma evidência mais forte que Fast93 porque não depende só de regra manual
ou interseção simples de sinais. Mesmo modelos não-lineares treinados em payoff
líquido e cauda ruim não conseguiram superar a instabilidade do regime.

O problema continua sendo estrutural:

1. o payoff por trade é fino;
2. custo e stress engolem a média;
3. o score não generaliza entre janelas;
4. quando reduzimos top fraction, o sistema fica silencioso;
5. quando aumentamos exposição, aparecem holdouts negativos grandes.

## Próxima direção

Não vale insistir em modelagem sobre o mesmo universo Fast92. O próximo caminho
precisa mudar a fonte de informação ou o regime:

1. shadow novo capturando features completas de livro, não só o vetor causal
   reduzido;
2. horizonte/payoff mínimo maior, buscando menos trades e margem maior;
3. segmentação de regime antes do modelo, não depois;
4. ou mudar de mercado/par/timeframe, porque BTC micro-horizon com este
   conjunto de features está instável.

## Validação

Comandos:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest \
  tests/unit/test_fast94_tail_risk_utility_model_audit.py \
  -q --no-cov
```

Resultado: `8 passed`.

```bash
./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast94_tail_risk_utility_model_audit.py \
  backend/tests/unit/test_fast94_tail_risk_utility_model_audit.py
```

Resultado: `All checks passed!`.

## Arquivos e hashes

- script:
  `backend/scripts/research/run_fast94_tail_risk_utility_model_audit.py`;
- teste:
  `backend/tests/unit/test_fast94_tail_risk_utility_model_audit.py`;
- report:
  `6cbc4e95dd0349bd438a987cf2eb45ba124a2146e55268ef21afb3c6bed19493`;
- candidate results:
  `4e4b07dd23288c90c2b9f7097eaf57e30cd4d436cfca19928adf78cf49720ecf`.
