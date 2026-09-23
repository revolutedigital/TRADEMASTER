# Fast73 — anti-stop + densidade

O diagnóstico do alvo mostrou um desalinhamento forte: o modelo atual acerta
`paid_stress_before_stop` com frequência, mas isso não garante lucro no trailing.
Fast73 testou uma hipótese diferente sobre o stream top-p congelado:

1. manter só sinais com `quote_volume_30s` alto, porque isso reduz initial STOP;
2. limitar densidade local para evitar clusters de sinais no mesmo surto.

Resultado: histórico bonito, mas falhou fora da amostra.

## Safety boundary

Fast73 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Alvo atual está desalinhado

Nos `2058` top-p históricos Fast43–Fast53:

- `paid_stress_before_stop` hit rate: `0.8809523809523809`;
- stress positivo real no trailing: `0.44946550048590866`;
- casos em que o alvo bateu, mas o trailing perdeu: `888`;
- `STOP` count: `170`;
- soma dos STOPs: `-3062.6277307829405` bps;
- soma dos não-STOPs: `+1237.4272276222382` bps.

Interpretação: existe payoff nos trades que não tomam initial stop, mas a
estratégia atual não prevê bem o STOP. O alvo "tocou antes do stop" é fraco
demais para trailing.

## Regra testada

- `quote_volume_30s >= 102102859.39293046`;
- threshold = q0.80 dos top-p históricos Fast43–Fast53;
- density: no máximo `2` sinais filtrados nos últimos `300s`.

## Resultado histórico Fast43–Fast53

- trades: `232`;
- blocos com amostra mínima: `10/11`;
- blocos suficientes positivos: `10`;
- blocos suficientes negativos: `0`;
- stress total: `593.7701748508641` bps;
- expected total: `825.7701748508641` bps;
- pior bloco suficiente: `18.543398904649766` bps.

## Resultado extensão Fast56 Sep21–Sep22

- trades: `15`;
- stress total: `-14.589625678347701` bps;
- expected total: `0.41037432165229615` bps;
- blocos suficientes negativos: `1`.

## Resultado Sep23 WAL

Sobre os settlements completos do top-p Sep23:

- top-p completos: `333`;
- após quote-volume: `34`;
- após density: `3`;
- stress independente: `-28.40732450520631` bps;
- one-position trades: `2`;
- one-position stress: `-10.40019844699578` bps.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast73-anti-stop-density-audit/report.json`;
- blocos históricos: `backend/data/microstructure_v1/reports/fast73-anti-stop-density-audit/historical_blocks.json`;
- extensão: `backend/data/microstructure_v1/reports/fast73-anti-stop-density-audit/extension_blocks.json`;
- Sep23: `backend/data/microstructure_v1/reports/fast73-anti-stop-density-audit/sep23_summary.json`.

Hashes:

- report: `d5536395a2a73484b934e4af791177f68970c6325ad9a6fc6ac288ca1ff74348`;
- historical blocks: `be1dbc5f5478c956525dd7f1dc2ed568f3e0499be4bb9f6c0725edf43f70bb42`;
- extension blocks: `4d97791b2a24e6f24c761ade36fbe0d2e58c7af523b0f0907c1ff1646937d602`;
- Sep23 summary: `e385b4b40c9b77a81ece81680e3848048ec9a290f438ef56bdb5be5c78aec385`.

## Decisão

Status:

`FAST73_ANTI_STOP_DENSITY_FAILED_EXTENSION`

Não aprovar estratégia. Não Testnet. Não LIVE.

O insight que fica não é a regra qv+density, porque ela falhou fora da amostra.
O insight útil é que a próxima família precisa mirar diretamente "não tomar
initial STOP / PnL de trailing", não `paid_stress_before_stop`.
