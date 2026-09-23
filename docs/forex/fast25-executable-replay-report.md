# Rodada 25 — replay executável por bid/ask da candidata Fast24

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast24 de 30 minutos  
Resultado: a candidata sobreviveu ao replay top-of-book até 1 bps de custo explícito; ainda não é estratégia aprovada

## Pergunta

Fast24 confirmou a candidata `spot_update_age_ms low 33%` usando path bruto de trades. A dúvida agora é
se o edge continua quando a entrada e a saída usam preço executável aproximado:

- SELL entra no `bid`;
- BUY de fechamento sai no `ask`;
- o spread bid/ask já entra no `gross_executable_bps`;
- depois ainda subtraí custo explícito de 0 / 0,25 / 0,5 / 1 / 2 bps;
- nenhuma ordem foi enviada.

Regra congelada:

```text
Base: SELL quando directed_spot_perp_basis_bps está no top 20% fitado em Fast16.
Gate: permitir só quando spot_update_age_ms está no bottom 33% fitado em Fast20+Fast21+Fast22.
Horizon: 120s.
Target: 5 bps.
Stop inicial: 20 bps.
```

## O que foi implementado

Script novo:

`backend/scripts/research/run_book_executable_replay.py`

Ele:

- reutiliza a seleção base+gate congelada;
- carrega os snapshots de book/depth do WAL;
- simula entrada marketable taker após latência configurável;
- calcula saída por target/stop/timeout no lado executável oposto;
- reporta dois modos:
  - `independent`: cada sinal é medido isoladamente;
  - `one_position`: ignora sinais enquanto uma posição hipotética anterior ainda está aberta.

Flags de segurança:

| Campo | Valor |
|---|---|
| `research_only` | `true` |
| `order_submission_allowed` | `false` |
| `execution_authorization` | `none` |
| `approved_strategy` | `false` |

## Resultado principal — 100ms de latência

| Modo | Sinais | Trades executáveis | Gross médio | Net 0,5 bps | Net 1 bps | Net 2 bps | Saídas |
|---|---:|---:|---:|---:|---:|---:|---|
| Independent | 52 | 52 | +1,340 bps | +0,840 bps | +0,340 bps | -0,660 bps | 20 TARGET / 32 TIMEOUT |
| One-position | 52 | 14 | +1,469 bps | +0,969 bps | +0,469 bps | -0,531 bps | 6 TARGET / 8 TIMEOUT |

Interpretação:

- O edge não desapareceu quando trocamos trade-last por bid/ask.
- O modo one-position também ficou positivo até 1 bps.
- Em 2 bps, o resultado de 100ms fica negativo.

## Sensibilidade de latência

| Latência | Independent net 1 bps | Independent net 2 bps | One-position trades | One-position net 1 bps | One-position net 2 bps |
|---:|---:|---:|---:|---:|---:|
| 0 ms | +0,346 bps | -0,654 bps | 14 | +0,469 bps | -0,531 bps |
| 100 ms | +0,340 bps | -0,660 bps | 14 | +0,469 bps | -0,531 bps |
| 250 ms | +0,370 bps | -0,630 bps | 13 | +1,087 bps | +0,087 bps |
| 500 ms | +0,350 bps | -0,650 bps | 13 | +1,087 bps | +0,087 bps |
| 1000 ms | +0,236 bps | -0,764 bps | 13 | +1,087 bps | +0,087 bps |

O resultado não quebrou com latência até 1000ms nessa janela. Mas isso não deve ser lido como prova de
robustez universal: a amostra one-position tem só 13-14 trades.

## Decisão

Temos uma candidata para shadow replay.

Ainda não temos estratégia aprovada para Testnet real nem LIVE.

O que melhorou:

- passou holdout futuro Fast24;
- passou replay bid/ask;
- passou com restrição one-position;
- ficou positivo em 1 bps de custo explícito;
- manteve flags fail-closed.

O que ainda bloqueia:

- margem ainda pequena;
- independent fica negativo em 2 bps;
- one-position tem N baixo;
- replay ainda não modela fila, liquidez por quantidade, partial fill, rejeição, funding/fees reais ou latência real medida;
- falta shadow replay contínuo com logs de sinais em tempo real;
- falta gate estatístico final.

Próximo passo correto:

1. transformar a regra em política shadow-only;
2. registrar sinais ao vivo sem ordem;
3. liquidar resultados pelo mesmo replay bid/ask;
4. só promover para Testnet se o shadow mantiver margem com custo realista.

Nenhuma ordem foi enviada.

## Artefatos

- Script replay executável:
  `backend/scripts/research/run_book_executable_replay.py`
- Testes:
  `backend/tests/unit/test_book_executable_replay.py`
- Replay 100ms:
  `backend/data/microstructure_v1/reports/book-executable-replay-fast24-spot-age-f20-f21-f22-to-f24/report.json`
  (`55eb5b802cf4376c08eb2a204237031df776492e2df473d1e9e59548d35a26e6`)
- Replay 0ms:
  `backend/data/microstructure_v1/reports/book-executable-replay-fast24-spot-age-latency-0ms/report.json`
  (`97a65f7002fce3113cc0fef08d1a7c3b022825e476beaf082965111186004064`)
- Replay 250ms:
  `backend/data/microstructure_v1/reports/book-executable-replay-fast24-spot-age-latency-250ms/report.json`
  (`c22e0828bbee712ffc041695142c59bcfdd0e91102586a67717012ef0f76bd93`)
- Replay 500ms:
  `backend/data/microstructure_v1/reports/book-executable-replay-fast24-spot-age-latency-500ms/report.json`
  (`1208d23a3c6f452603d93d2d507c8f60ce324f5567aaae610a38c95e747a0b63`)
- Replay 1000ms:
  `backend/data/microstructure_v1/reports/book-executable-replay-fast24-spot-age-latency-1000ms/report.json`
  (`062e67b572294736f38250d5012d3244da13f2948672cc32ca237bf42cbc0c56`)
