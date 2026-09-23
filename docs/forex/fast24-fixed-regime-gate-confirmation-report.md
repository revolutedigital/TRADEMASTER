# Rodada 24 — holdout futuro da candidata `spot_update_age_ms`

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast24 de 30 minutos  
Resultado: a candidata robusta passou em uma janela futura; ainda não está aprovada para operar

## Pergunta

Fast23 escolheu uma candidata por estabilidade walk-forward, não por top-1:

```text
Base: SELL quando directed_spot_perp_basis_bps está no top 20% fitado em Fast16.
Gate: permitir só quando spot_update_age_ms está no bottom 33% fitado em Fast20+Fast21+Fast22.
Horizonte: 120s.
Alvo bruto: 5 bps.
```

Fast24 é o primeiro holdout novo depois dessa escolha. A regra não foi reotimizada com Fast24.

## Captura Fast24

Comando:

`PYTHONPATH=. .venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-fast24-30m --symbol BTCUSDT --duration-seconds 1800 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-fast24-30m-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 120`

Streams capturados:

| Stream | Linhas | Início UTC | Fim UTC |
|---|---:|---|---|
| `trade` | 99.318 | 2026-09-23T05:21:13.435Z | 2026-09-23T05:51:12.435Z |
| `depth` | 17.636 | 2026-09-23T05:21:13.702Z | 2026-09-23T05:51:12.370Z |
| `mark_price` | 1.381 | 2026-09-23T05:21:12Z | 2026-09-23T05:51:11.006Z |
| `spot_trade` | 70.371 | 2026-09-23T05:21:15.206Z | 2026-09-23T05:51:12.796Z |

Scout WAL:

| Campo | Valor |
|---|---:|
| Status | `READY_FOR_SCOUT_MODELING` |
| Decisões | 330 |
| Linhas | 660 |

## Resultado da regra base sem gate

| Campo | Valor |
|---|---:|
| Seleções | 166 |
| Payoff bruto selecionado | +1,098 bps |
| Target rate selecionado | 37,35% |
| Net com 0,5 bps | +0,598 bps |
| Net com 1,0 bps | +0,098 bps |
| Net com 2,0 bps | -0,902 bps |

A regra base também foi positiva em Fast24, mas por margem estreita em 1 bps e negativa em 2 bps.

## Resultado com gate fixo

Gate fitado em Fast20+Fast21+Fast22:

| Campo | Valor |
|---|---:|
| Feature | `spot_update_age_ms` |
| Direção | low |
| Fração | 33% |
| Threshold congelado | 140,48 ms |

Holdout Fast24:

| Campo | Valor |
|---|---:|
| Status | `FIXED_GATE_CONFIRMATION_READY_FOR_REPLICATION` |
| Seleções | 52 |
| Payoff bruto selecionado | +1,564 bps |
| Target rate selecionado | 40,38% |
| Lift bruto vs base geral | +2,058 bps |
| Net com 0,5 bps | +1,064 bps |
| Net com 1,0 bps | +0,564 bps |
| Net com 2,0 bps | -0,436 bps |

O gate melhorou a regra base em Fast24: menos trades, payoff maior e margem melhor contra 1 bps de custo.

## Controle: top-1 do desenvolvimento falhou

Também rodei o scout usando Fast20+Fast21+Fast22 como desenvolvimento e Fast24 como confirmação.
O top-1 escolhido pelo desenvolvimento foi:

| Campo | Valor |
|---|---:|
| Gate top-1 | `trade_count_30s` high 10% |
| Seleções em Fast24 | 13 |
| Gross em Fast24 | -0,929 bps |
| Net com 1,0 bps | -1,929 bps |

Isso reforça a lição principal: perseguir o top-1 do treino continua perigoso. A candidata que passou
foi a selecionada por estabilidade entre folds.

## Decisão

Agora temos uma candidata séria de pesquisa:

- passou em 3/3 folds internos de Fast20/Fast21/Fast22;
- passou em Fast24, uma janela futura após a seleção;
- em Fast24 gerou 52 seleções;
- ficou positiva até 1 bps de custo total;
- melhorou a regra base pura.

Ainda não é estratégia aprovada para bot operar dinheiro.

Bloqueios restantes:

- a 2 bps de custo total ela ainda fica negativa em Fast24;
- falta replay/simulação de execução maker/taker, fila, latência e slippage;
- falta medir estabilidade em mais janelas e em outros regimes;
- falta gate estatístico final;
- `approved_strategy=false`, `order_submission_allowed=false`, `execution_authorization=none`.

Próximo passo correto:

1. construir replay de execução sem envio de ordem;
2. medir custo realista maker/taker/slippage;
3. rodar mais uma janela prospectiva se o custo real passar de 1 bps;
4. só depois considerar Testnet shadow.

Nenhuma ordem foi enviada. Demo/Testnet/LIVE seguem bloqueados.

## Artefatos

- WAL scout Fast24:
  `backend/data/microstructure_v1/reports/book-wal-scout-fast24-30m/report.json`
  (`747ceda3be8b408977cdf546d3dec1d71d924fe37917f8832755c386aa353a10`)
- Confirmação fixa Fast24:
  `backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast24-spot-age-f20-f21-f22-to-f24/report.json`
  (`e165fc78aae82281dcbc56d644924a1d19a1512d4369c7171adc8d20a5e13abe`)
- Scout top-1 Fast20+Fast21+Fast22 → Fast24:
  `backend/data/microstructure_v1/reports/book-regime-gate-scout-fast24-spot-f20-f21-f22-to-f24/report.json`
  (`6fa84d10806aab779747074c365b9d8ec2b852f83f6b8624f406e6094b5acf33`)
- Audit status Fast24:
  `backend/data/microstructure_v1/prospective-wal-fast24-30m-audits/evidence-gate-status.json`
  (`4df4f9a543d2b2554f8b25d3f5da37eafc6158ef5f829a22d855b89f953ce97a`)
