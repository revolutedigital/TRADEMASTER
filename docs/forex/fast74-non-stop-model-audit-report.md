# Fast74 — modelo de probabilidade de não tomar STOP

Resposta objetiva: ainda não temos estratégia aprovada.

Fast74 testou a hipótese que parecia mais alinhada com o que queremos para
trailing stop: em vez de prever `paid_stress_before_stop`, treinar um modelo
para prever se o trade evita `first_touch == STOP`.

O resultado explica por que a ideia parecia boa e por que ainda não é estratégia.

## Safety boundary

Fast74 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Setup testado

- base: top-p histórico Fast43–Fast53;
- trades históricos: `2058`;
- target: `first_touch != STOP`;
- features causais: `34`;
- modelo: `LogisticRegression`, `class_weight=balanced`;
- filtro de liquidez: `quote_volume_30s >= 102102859.39293046`;
- score congelado: `non_stop_probability >= 0.7698192709463426`;
- score tail: top `10%` da distribuição histórica de treino;
- density: no máximo `3` sinais filtrados nos últimos `300s`.

## Por que parecia estratégia

O ranking diagnóstico, usando o top 10% dentro de cada bloco já observado,
ficou forte:

- trades: `115`;
- stress total: `418.3334871023731` bps;
- expected total: `533.333487102373` bps;
- blocos suficientes: `8`;
- blocos suficientes positivos: `8`;
- pior bloco suficiente: `7.124816584323554` bps;
- blocos suficientes negativos: `0`.

Mas esse ranking não é executável como política congelada, porque usa a
distribuição de score do próprio bloco avaliado para escolher o top 10%.
Isso é útil como scout, não como bot.

## Teste honesto no histórico

Quando o cutoff de score vem só do treino anterior, sem usar a distribuição do
holdout:

- trades: `109`;
- stress total: `395.33966344393434` bps;
- expected total: `504.3396634439344` bps;
- blocos suficientes: `8`;
- blocos suficientes positivos: `7`;
- pior bloco suficiente: `-26.478667723961202` bps;
- bloco negativo: `latency-adjusted-frozen-top-p-fast50-lowcost-micro-holdout-aug25-aug31-100ms`.

Ainda existe sinal agregado, mas já aparece fragilidade.

## Teste fora-da-amostra Fast56 Sep21–Sep22

Com o modelo treinado em todo Fast43–Fast53 e thresholds congelados:

- trades: `8`;
- stress total: `-18.232404556319942` bps;
- expected total: `-10.232404556319942` bps;
- blocos suficientes: `1`;
- blocos suficientes positivos: `0`;
- bloco negativo: `latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms`.

Esse resultado mata a promoção para shadow candidato.

## Sep23 WAL

O Sep23 WAL atual não consegue scorear esse modelo completo:

- decisões raw top-p: `550`;
- settlements completos: `333`;
- features exigidas pelo modelo: `34`;
- features guardadas no WAL: `3`;
- features ausentes: `31`.

O WAL guardou apenas:

- `directed_flow_imbalance_30s`;
- `quote_volume_30s`;
- `realized_vol_30s_bps`.

Para um shadow real dessa família, o bot precisa registrar o vetor causal
completo usado no score, não só os valores do gate direcional.

## Arquivos

- script: `backend/scripts/research/run_fast74_non_stop_model_audit.py`;
- relatório JSON: `backend/data/microstructure_v1/reports/fast74-non-stop-model-audit/report.json`;
- histórico honesto: `backend/data/microstructure_v1/reports/fast74-non-stop-model-audit/honest_historical_blocks.json`;
- ranking diagnóstico: `backend/data/microstructure_v1/reports/fast74-non-stop-model-audit/diagnostic_ranked_historical_blocks.json`;
- extensão: `backend/data/microstructure_v1/reports/fast74-non-stop-model-audit/extension_blocks.json`;
- Sep23 scoreability: `backend/data/microstructure_v1/reports/fast74-non-stop-model-audit/sep23_scoreability.json`.

Hashes:

- report: `f3a2fb9f827661fa3f8bd123dc08055a2bd44b102ededfda93414eb6adbc8c48`;
- honest historical blocks: `1e45ffe87bd58031832856e0d42a7eede930682dd6b67b46fca49cc507a4b975`;
- diagnostic ranked blocks: `f711bbadf8bafccaf20d76530a6af620f5a827cfa1ea896bbd9353892670b57d`;
- extension blocks: `4288b7c70e60411147d93d3c5e4abd2b5d22b0ae6fb64d86a7328cae0abb00b5`;
- Sep23 scoreability: `c41af3ac50e6eaa31e866abe490269c619c23597ce5fc874f3bbbb496b0e1ae5`.

## Decisão

Status:

`FAST74_NON_STOP_MODEL_FAILED_EXTENSION`

Não aprovar estratégia. Não Testnet. Não LIVE.

O insight útil permanece: a direção certa é prever diretamente risco de initial
STOP / payoff do trailing, mas precisamos fazer isso em uma base de candidatos
mais ampla e com feature capture executável no shadow. O filtro sobre top-p
antigo ainda herda fragilidade demais.
