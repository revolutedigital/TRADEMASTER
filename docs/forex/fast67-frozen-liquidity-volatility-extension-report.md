# Fast67 — gate congelado de liquidez + volatilidade

Fast66 apontou a melhor família até aqui: operar o top-p congelado só quando
`quote_volume_30s` e `realized_vol_30s_bps` estão altos. Fast67 remove a última
ambiguidade: transforma os quantis em thresholds numéricos fixos calibrados em
Fast43–Fast53 e aplica a mesma régua em extensão posterior.

Resultado: o candidato não sobreviveu.

## Safety boundary

Fast67 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Regra congelada

Fonte de calibração: trades selecionados Fast43–Fast53 com features causais do
Fast65.

Gate:

- `quote_volume_30s >= 102102859.39293046`;
- `realized_vol_30s_bps >= 32.80573757966689`.

Isso corresponde a:

- `quote_volume_30s >= q0.80`;
- `realized_vol_30s_bps >= q0.75`.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast67-frozen-liquidity-volatility-extension/report.json`;
- gate congelado: `backend/data/microstructure_v1/reports/fast67-frozen-liquidity-volatility-extension/frozen_gate.json`;
- blocos históricos: `backend/data/microstructure_v1/reports/fast67-frozen-liquidity-volatility-extension/historical_blocks.json`;
- blocos de extensão: `backend/data/microstructure_v1/reports/fast67-frozen-liquidity-volatility-extension/extension_blocks.json`.

Hashes:

- report: `b55086a0e5d3e55a599ea9fdcd6f63df58c024f8727de0104ee793b8e3a79ca7`;
- frozen gate: `ca58233f38aa4dd3858fe585b8cc9a2bf338b3832467b608d6f8fd753366a9bd`;
- historical blocks: `50d04d5347d9e8c59bf4ef17ddc51a3a6385893d699ce2a50e150817d95245e5`;
- extension blocks: `f7c330647b9129b4fa84b9a121f4569aed933e2df4ca539d1bf359e212ffb8fd`.

## Resultado

Status:

`FAST67_FROZEN_GATE_FAILED_EXTENSION`

Histórico Fast43–Fast53:

- trades: `305`;
- blocos com amostra mínima: `10/11`;
- blocos positivos em stress: `9`;
- stress total: `469.4327997381948` bps;
- expected total: `774.4327997381947` bps;
- bloco suficiente negativo: Fast53 com `-3.9827702159693334` bps.

Extensão Fast56 Sep21–Sep22:

- trades: `26`;
- blocos positivos em stress: `0/1`;
- stress total: `-38.985635198338166` bps;
- expected total: `-12.985635198338166` bps;
- bloco negativo suficiente:
  `latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms`.

## Interpretação

O agregado histórico continuava bonito, mas isso não basta. Quando o gate deixa
de ser uma família promissora e vira número fixo, ele:

- ainda carrega um bloco histórico negativo suficiente;
- falha no primeiro bloco de extensão disponível;
- prova que “mercado com liquidez + volatilidade alta” também captura mercado
em pânico, onde o modelo pode entrar contra uma pernada forte.

Portanto, não há estratégia aprovada. Não Testnet. Não LIVE.

## Próxima hipótese

O fracasso de Sep21–Sep22 concentrou sinais SELL com retorno recente positivo.
O próximo teste deve ser uma nova família, pré-registrada separadamente, que
adicione alinhamento direcional causal ao gate de liquidez/volatilidade. Isso
não pode ser tratado como ajuste do Fast67 depois de ver o fracasso; é Fast68.
