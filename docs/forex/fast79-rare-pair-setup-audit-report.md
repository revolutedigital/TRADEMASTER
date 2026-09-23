# Fast79 — scout de setup raro com duas condições causais

Resposta objetiva: ainda não temos estratégia.

Fast79 testou uma hipótese nova em cima do stream top-p histórico: talvez a
entrada só funcione em um setup raro de microestrutura, definido por duas
condições causais simples, em vez de um gate amplo de liquidez/volatilidade.

Este é um scout em etapas, não uma prova exaustiva:

1. gerar condições atômicas em features causais pré-trade;
2. escolher os melhores átomos só pelo histórico Fast43–Fast53;
3. testar pares desses átomos no histórico;
4. auditar os melhores pares no Fast56 Sep21–Sep22 sem usar Fast56 para
   selecionar.

## Safety boundary

Fast79 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Fonte

Histórico:

`backend/data/microstructure_v1/reports/fast65-causal-feature-gate-walkforward/selected_trades_with_causal_features.parquet`

Extensão:

`backend/data/microstructure_v1/reports/latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms/report.json`

## Busca

- features causais: `34`;
- condições atômicas: `340`;
- átomos elegíveis: `340`;
- pares avaliados: `1770`;
- pares elegíveis: `1629`;
- sobreviventes históricos: `11`;
- pares auditados na extensão: `25`;
- sobreviventes positivos na extensão: `6`.

Status:

`FAST79_RARE_PAIR_SETUP_WEAK_SURVIVORS_REQUIRE_PREREGISTRATION`

## Melhor por stress na extensão

Regra:

`realized_vol_30s_bps>=q0.80:45.38807193 AND flow_imbalance_5s<=q0.33:-0.219307208`

Histórico Fast43–Fast53:

- one-position trades: `138`;
- stress total: `+418.2238564179386` bps;
- stress médio: `+3.0306076552024535` bps;
- blocos suficientes positivos: `10/10`;
- pior bloco suficiente: `+9.334692784221659` bps.

Extensão Fast56:

- one-position trades: `3`;
- stress total: `+28.635546270573894` bps;
- stress médio: `+9.545182090191298` bps;
- dias distintos: `1`.

Interpretação: o resultado da extensão é bom, mas a amostra é pequena demais
para promoção. Três trades podem ser regime/sorte.

## Melhor com mais trades na extensão

Regra:

`flow_imbalance_60s<=q0.33:-0.2155063064 AND directed_flow_imbalance_60s<=q0.20:-0.4722264655`

Histórico Fast43–Fast53:

- one-position trades: `194`;
- stress total: `+543.5754885215758` bps;
- stress médio: `+2.8019355078431745` bps;
- blocos suficientes positivos: `10/10`;
- pior bloco suficiente: `+23.32119418103448` bps.

Extensão Fast56:

- one-position trades: `11`;
- stress total: `+0.3685482970949798` bps;
- stress médio: `+0.03350439064499817` bps;
- dias distintos: `2`.

Interpretação: tem mais amostra, mas a margem depois de custo stress é
praticamente zero. Isso não aguenta erro de execução.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O Fast79 muda a leitura da pesquisa: existe uma família rara que recupera o
histórico e não quebra imediatamente no Fast56, mas ela ainda é frágil demais
para operação.

Próximo passo válido:

1. escolher uma única regra, com preferência por robustez operacional e não pelo
   maior resultado de 3 trades;
2. pré-registrar a regra antes de qualquer novo dado;
3. garantir que o shadow capture todas as features exigidas pela regra;
4. rodar prospectivamente em WAL limpo;
5. exigir amostra mínima e margem stress positiva antes de qualquer discussão de
   shadow offline/Testnet.

## Arquivos

- script: `backend/scripts/research/run_fast79_rare_pair_setup_audit.py`;
- teste: `backend/tests/unit/test_fast79_rare_pair_setup_audit.py`;
- relatório JSON:
  `backend/data/microstructure_v1/reports/fast79-rare-pair-setup-audit/report.json`;
- avaliações atômicas:
  `backend/data/microstructure_v1/reports/fast79-rare-pair-setup-audit/atom_evaluations.json`;
- avaliações de pares:
  `backend/data/microstructure_v1/reports/fast79-rare-pair-setup-audit/pair_evaluations.json`;
- auditoria de extensão:
  `backend/data/microstructure_v1/reports/fast79-rare-pair-setup-audit/extension_audits.json`.

Hashes:

- report: `bc9b8707e5bb8980b96aac7cb9db306fab4fb98bab83dfff16130a60cfaea7ba`;
- atom evaluations:
  `c01fd0a176128448f76fd7bcb889428e92f90e5798ef757b4a42cb9599cd04f6`;
- pair evaluations:
  `7f8fb800aaa5e4b203a7919e82f24078af81000e7d3f64b782fe9d8577c8b071`;
- extension audits:
  `181ddf1261bdfbe4c948b34483ad3b567ca69a9354a049bfcd8f8b3968ae226e`.
