# Fast69 — shadow dry-run do gate direcional congelado

Fast69 transforma o melhor survivor diagnóstico do Fast68 em artifact executável
de pesquisa e roda o gate em WAL local, sem reabrir busca e sem enviar ordens.

Resultado: o gate executável gerou sinais, mas falhou no settlement conhecido.

## Safety boundary

Fast69 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Artifact congelado

Arquivo:

`backend/data/microstructure_v1/models/fast69-directional-liquidity-volatility-shadow-policy.json`

Hash:

`56d84d8ea4a1917e2e127baf40dd881990cc0401951a671221cc5e093b6e8e6b`

Base top-p:

- model sha: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- threshold: `0.90195311025804`;
- tail fraction: `0.01`;
- horizon: `300s`.

Gate Fast68 congelado:

- `quote_volume_30s >= 79775575.91570444`;
- `realized_vol_30s_bps >= 45.38807193200911`;
- `realized_vol_30s_bps <= 351.4988811749231`;
- `directed_flow_imbalance_30s >= -0.4391006273798949`.

## Dry-run WAL

Relatório:

`backend/data/microstructure_v1/reports/fast69-directional-shadow-dry-run/report.json`

Hash:

`5643b36b76e1033a1239098d4180366f0493f306795717ec8e40cde939630ea3`

Resultado:

- top-p bruto: `550` sinais;
- sinais após gate direcional: `19`;
- runs com sinais: `1/10`;
- status: `FAST69_DIRECTIONAL_SHADOW_SIGNALS_RESEARCH_ONLY`.

Hashes:

- raw decisions: `e21ce0690dca5826cc0dc7a70e610f5a29571f4b9d4bbb8b452b88dfea9b59f2`;
- gated decisions: `93000de6d3395ba837e96c0ff157239253e6cc13863cfefa668a762e26557004`.

## Settlement Fast70

O settlement foi rodado sobre as 19 decisões do Fast69 com o replay/trailing
existente.

Relatório:

`backend/data/microstructure_v1/reports/fast70-fast69-directional-shadow-settlement/report.json`

Hash:

`9e76602d09dac8fe86a3464427d68089da7f203e746c4a2c3abe74e6099814dd`

Resultado independente:

- trades completos: `19`;
- stress total: `-57.76378806944887` bps;
- stress mean: `-3.0401993720762563` bps;
- expected total: `-38.76378806944887` bps.

Resultado one-position:

- trades: `6`;
- stress total: `-21.078773321887045` bps;
- stress mean: `-3.5131288869811743` bps;
- expected total: `-15.078773321887045` bps.

Status:

`FAST61_SHADOW_SETTLEMENT_FAILED`

## Interpretação

O gate direcional reduziu bastante a exposição, mas não resolveu o problema.
Pior: no online, o corte `directed_flow_imbalance_30s >= -0.439...` ainda é
permissivo demais. Ele permite situações ambíguas, inclusive sinais BUY e SELL
no mesmo timestamp quando ambos passam no top-p.

Conclusão: Fast69 não é estratégia. Não Testnet. Não LIVE.

Próxima hipótese: uma família direcional mais estrita, com alinhamento
positivo de fluxo/retorno e bloqueio explícito de timestamps onde BUY e SELL
passam juntos.
