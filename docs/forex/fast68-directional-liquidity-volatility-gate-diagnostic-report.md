# Fast68 — gate direcional de liquidez + volatilidade

Fast67 matou o gate puro de liquidez + volatilidade. O fracasso de extensão
Sep21–Sep22 concentrou entradas SELL contra retorno/fluxo recente de alta. O
Fast68 testa uma nova família: só operar quando a microestrutura está ativa,
mas com alinhamento direcional causal antes da entrada.

Resultado: apareceram sobreviventes diagnósticos, mas ainda não há estratégia
aprovada.

## Safety boundary

Fast68 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Escopo

- fonte histórica: trades Fast43–Fast53 mergeados com features causais do
  Fast65;
- extensão: Fast56 Sep21–Sep22;
- execução: one-position;
- custo: `stress_net_bps` materializado nos trades congelados;
- busca: 2.304 combinações;
- critério survivor:
  - nenhum bloco histórico com amostra suficiente negativo;
  - pelo menos 5 blocos históricos com amostra suficiente;
  - extensão com amostra suficiente e stress positivo.

Importante: a extensão participou da seleção diagnóstica. Portanto, qualquer
survivor precisa virar hipótese congelada para um shadow futuro limpo.

## Melhor survivor

Spec:

`quote_volume_30s>=q0.70 AND realized_vol_30s_bps>=q0.80 AND realized_vol_30s_bps<=q0.99 AND directed_flow_imbalance_30s>=q0.33`

Thresholds congelados a partir de Fast43–Fast53:

- `quote_volume_30s >= 79775575.91570444`;
- `realized_vol_30s_bps >= 45.38807193200911`;
- `realized_vol_30s_bps <= 351.4988811749231`;
- `directed_flow_imbalance_30s >= -0.4391006273798949`.

Resultado histórico walk-forward Fast43–Fast53:

- trades: `147`;
- blocos com amostra mínima: `8/11`;
- blocos suficientes positivos: `8`;
- blocos suficientes negativos: `0`;
- stress total: `230.02788113386907` bps;
- expected total: `377.0278811338691` bps.

Resultado extensão Fast56 Sep21–Sep22:

- trades: `14`;
- blocos suficientes positivos: `1/1`;
- stress total: `11.275780282481207` bps;
- expected total: `25.275780282481207` bps.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast68-directional-gate-diagnostic/report.json`;
- top candidates: `backend/data/microstructure_v1/reports/fast68-directional-gate-diagnostic/top_candidates.json`;
- survivors: `backend/data/microstructure_v1/reports/fast68-directional-gate-diagnostic/survivors.json`.

Hashes:

- report: `726e9a0c6e7fe1d5bb5bbc0eb268f4350b21088486327725c4fbaee4ed632630`;
- top candidates: `c6d69d13e4d0150ed1fba93a668a2f9ffa1c719722f0d0efbb3c3a51c224b8aa`;
- survivors: `c94604af2001a847b9af9cecea27b9f26dcef8f200618674c833559f8948a7ba`.

## Decisão

Status:

`FAST68_DIRECTIONAL_GATE_DIAGNOSTIC_SURVIVORS_FOUND_REQUIRES_SHADOW`

Não aprovar estratégia. Não Testnet. Não LIVE.

O melhor candidato agora merece pré-registro/shadow limpo:

1. congelar exatamente a spec e os thresholds acima;
2. rodar só em dados futuros/prospectivos não usados no Fast68;
3. exigir amostra mínima antes de qualquer conclusão;
4. reprovar se qualquer bloco novo suficiente ficar negativo em stress;
5. só depois discutir Testnet, ainda sem LIVE.
