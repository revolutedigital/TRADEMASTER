# Fast71/Fast72 — diagnóstico direcional estrito

Depois que o Fast69 falhou, foi testado um diagnóstico mais duro sobre os
mesmos sinais top-p já materializados:

- manter thresholds de liquidez/volatilidade do Fast69;
- exigir `directed_flow_imbalance_30s >= 0`;
- remover timestamps onde BUY e SELL passam juntos.

Isso é diagnóstico pós-falha, não pré-registro e não candidato de estratégia.

## Fast71 — filtro

Relatório:

`backend/data/microstructure_v1/reports/fast71-strict-directional-filter-diagnostic/report.json`

Hash:

`8fdb8de343c97f8099607a15db102de57af020abedb33e7de12bec89a71b4275`

Resultado:

- top-p bruto: `550`;
- sinais após filtro estrito: `12`;
- status: `FAST71_STRICT_DIRECTIONAL_FILTER_DIAGNOSTIC_SIGNALS`;
- gated decisions hash:
  `87a6ec018d11a59ee5c4436f872f314d0f4b78e7c027ad90d78ae0b28f87d2fb`.

## Fast72 — settlement

Relatório:

`backend/data/microstructure_v1/reports/fast72-fast71-strict-directional-settlement/report.json`

Hash:

`7d24d7ba6b8380dab7601e5e32e008b556e5038a4d735d2bf999215a88eafd3e`

Resultado independente:

- trades: `12`;
- stress total: `-68.4170642708406` bps;
- stress mean: `-5.70142202257005` bps;
- expected total: `-56.417064270840605` bps.

Resultado one-position:

- trades: `5`;
- stress total: `-12.505056758317366` bps;
- stress mean: `-2.501011351663473` bps;
- expected total: `-7.505056758317366` bps.

Status:

`FAST61_SHADOW_SETTLEMENT_FAILED`

## Decisão

Não aprovar estratégia. Não Testnet. Não LIVE.

O diagnóstico fecha a hipótese de que bastaria apertar o alinhamento direcional
simples por fluxo 30s. Se houver próxima rodada, ela precisa testar uma família
diferente, por exemplo:

- bloqueio explícito de regime de notícia/pânico;
- confirmação multi-horizonte com assimetria forte entre BUY e SELL;
- ou reduzir ainda mais o universo para setup raro com MFE/MAE estruturalmente
favorável antes do trailing.
