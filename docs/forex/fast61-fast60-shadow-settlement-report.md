# Fast61 — settlement dos sinais Fast60

Fast61 fecha a pergunta deixada pelo Fast60:

> os sinais live-like emitidos pelo candidato Fast57 ganhariam dinheiro depois
> de trailing, latência e custo?

Resultado: **não**. O candidato continua como pesquisa reprovada, sem Testnet e
sem LIVE.

## Escopo

- entrada: `fast60-fast57-shadow-dry-run/gated_shadow_decisions.jsonl`;
- candidato: Fast57 / modelo `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- regra: top-p 1%, threshold `0.90195311025804`, gate `recent_300s_lte_0`;
- política: `micro_8_3`;
- horizonte: 300s;
- latência: 100ms;
- custo esperado/stress: 1 / 2 bps roundtrip;
- liquidação: replay causal de trades do WAL, sem enviar ordens.

## Safety boundary

O relatório mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Resultado

Arquivo principal:

- `backend/data/microstructure_v1/reports/fast61-fast60-shadow-settlement/report.json`

Hashes:

- report: `7f5965f6cde4a9d3f3115409e6d59b6539432754f64a7b3ff2b57e8d715d4839`;
- independent settlements: `b33a1bae5f2a28c32a43250993a5538f28e820624a39d5100c835958157f6bd8`;
- one-position settlements: `b33a1bae5f2a28c32a43250993a5538f28e820624a39d5100c835958157f6bd8`;
- incomplete signals: `d21633f892ad93891f9d7b3a5dc59903248f3e89ec71fd3a4945fcd9d063d152`.

Resumo:

- sinais Fast60 gated: 11;
- sinais com horizonte completo: 8;
- sinais incompletos: 3;
- dias de decisão completos: 1;
- status: `FAST61_SHADOW_SETTLEMENT_FAILED`.

PnL:

- independent stress mean: `-5.078671963738999` bps;
- independent stress total: `-40.62937570991199` bps;
- one-position stress mean: `-5.078671963738999` bps;
- one-position stress total: `-40.62937570991199` bps;
- expected win-rate: `0.625`;
- max drawdown esperado: `44.86345625010173` bps.

Como não houve sobreposição entre os 8 sinais completos, `independent` e
`one-position` ficaram iguais.

## Diagnóstico

O resultado reprova a hipótese de que “boa entrada + trailing” bastaria para
transformar o candidato em estratégia. A amostra completa teve 5/8 trades com
resultado esperado positivo, mas o stress ficou negativo porque:

- alguns ganhos foram pequenos demais para pagar 2 bps;
- dois stops grandes dominaram a distribuição;
- três sinais não tinham horizonte completo no WAL e foram corretamente
  excluídos, não contados como vitória.

Trades completos:

| decision_time UTC | side | gross bps | stress bps | exit |
| --- | --- | ---: | ---: | --- |
| 2026-09-23T03:40:30 | SELL | 1.9918 | -0.0082 | BREAKEVEN_STOP |
| 2026-09-23T04:21:05 | SELL | -16.0057 | -18.0057 | INITIAL_STOP |
| 2026-09-23T04:34:25 | BUY | -15.5227 | -17.5227 | TIMEOUT |
| 2026-09-23T05:22:15 | BUY | 4.3692 | 2.3692 | TIMEOUT |
| 2026-09-23T11:26:05 | BUY | 3.2045 | 1.2045 | TIMEOUT |
| 2026-09-23T12:03:15 | BUY | 1.0996 | -0.9004 | TIMEOUT |
| 2026-09-23T13:30:55 | SELL | -16.0084 | -18.0084 | INITIAL_STOP |
| 2026-09-23T13:53:15 | BUY | 12.2423 | 10.2423 | BREAKEVEN_STOP |

Sinais incompletos:

- 2026-09-23T05:49:40 BUY — `horizon_not_fully_covered_by_wal`;
- 2026-09-23T12:17:20 BUY — `horizon_not_fully_covered_by_wal`;
- 2026-09-23T14:00:20 SELL — `horizon_not_fully_covered_by_wal`.

## Conclusão

Não existe estratégia aprovada aqui. Existe uma pipeline de descoberta e um
candidato Fast57 que parecia interessante em seed/prospectivo parcial, mas
falhou quando os sinais live-like foram liquidados contra caminho de preço real.

Próximo caminho correto: não promover Fast57. A próxima busca deve atacar a
distribuição de payoff, não só a probabilidade de acertar direção: exigir
potencial de deslocamento suficiente para pagar custo, filtrar stops grandes e
validar em shadow settlement antes de qualquer Testnet.
