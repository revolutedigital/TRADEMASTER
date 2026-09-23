# Rodada 16 — scout prospectivo com book/depth fresco

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo público  
Resultado: pipeline com book aprovado para scout; estratégia ainda não encontrada

## Pergunta

As rodadas 12 a 15 rejeitaram variações de top-p, breakeven/trailing, latência e ranking por payoff
usando apenas preço/volume agregado. A hipótese restante era que o gargalo podia estar na informação
disponível: sem book, spread real, microprice, mark e relação spot-perp, o modelo ficava tentando
extrair sinal de uma camada pobre demais.

Esta rodada não tenta aprovar robô. Ela materializa um WAL prospectivo curto com `trade + depth`
e features causais de book para dizer se já existe base limpa suficiente para os próximos scouts.

## O que foi implementado

Script novo:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_wal_scout.py --wal-root <wal> --output-root <out> --horizons-seconds 30 --decision-stride-seconds 5 --minimum-decisions-for-modeling 200`

O script:

- exige `trade` e `depth`;
- usa book fresco com `max_book_staleness_ms=1000`;
- cria decisões a cada 5s, respeitando janela de histórico e horizonte futuro;
- chama `build_research_rows` com `book_events`, `mark_events` e `spot_trade_events` quando disponíveis;
- grava `research_rows.parquet` e `report.json`;
- falha fechado quando a captura é curta demais;
- mantém flags explícitas de pesquisa: `research_only=true`, `order_submission_allowed=false`,
  `execution_authorization=none`.

## Smoke anterior: v9

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_wal_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-smoke-v9 --output-root backend/data/microstructure_v1/reports/book-wal-scout-v9 --horizons-seconds 30 --decision-stride-seconds 5 --minimum-decisions-for-modeling 200`

Resultado:

| Campo | Valor |
|---|---:|
| Status | `CAPTURE_TOO_SHORT` |
| Decisões | 6 |
| Linhas | 12 |
| Book disponível | 100% |
| Máx. idade do book | 88 ms |
| Mark features | sim |
| Spot-perp features | sim |

O v9 provou que o parser funcionava, mas não tinha amostra suficiente para modelagem.

## Captura prospectiva Fast16 — 18 minutos

Comando:

`PYTHONPATH=. .venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-fast16-18m --symbol BTCUSDT --duration-seconds 1080 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-fast16-18m-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 120`

Streams capturados:

| Stream | Linhas | Início UTC | Fim UTC |
|---|---:|---|---|
| `trade` | 21.737 | 2026-09-23T02:50:51.601Z | 2026-09-23T03:08:50.298Z |
| `depth` | 10.578 | 2026-09-23T02:50:51.466Z | 2026-09-23T03:08:50.217Z |
| `mark_price` | 829 | 2026-09-23T02:50:50Z | 2026-09-23T03:08:50.002Z |
| `spot_trade` | 18.726 | 2026-09-23T02:50:51.703Z | 2026-09-23T03:08:51.641Z |

Scout:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_wal_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --output-root backend/data/microstructure_v1/reports/book-wal-scout-fast16-18m --horizons-seconds 30 --decision-stride-seconds 5 --minimum-decisions-for-modeling 200`

Resultado:

| Campo | Valor |
|---|---:|
| Status | `READY_FOR_SCOUT_MODELING` |
| Decisões | 204 |
| Linhas | 408 |
| Book disponível | 100% |
| Idade média do book | 51,08 ms |
| Máx. idade do book | 103 ms |
| Mark features | sim |
| Spot-perp features | sim |

## Resultado dos alvos de 30s

Configuração do dataset:

| Campo | Valor |
|---|---:|
| Horizonte | 30s |
| Custo esperado | 12 bps |
| Custo stress | 24 bps |
| Stop inicial | 20 bps |

Resultado por lado:

| Lado | Linhas | Target expected | Target stress | Terminal médio | MFE médio | MAE médio |
|---|---:|---:|---:|---:|---:|---:|
| BUY | 204 | 0,00% | 0,00% | -0,061 bps | 0,869 bps | -0,902 bps |
| SELL | 204 | 0,00% | 0,00% | 0,061 bps | 0,902 bps | -0,869 bps |

Leitura: nesse recorte calmo, o preço até mexeu, mas mexeu muito menos do que o alvo líquido de
12/24 bps. O trailing stop não resolve isso sozinho; se a entrada não anda o suficiente antes do
ruído, o trailing só reduz perdas ou transforma trades em stop/zero-a-zero, sem criar edge.

## Decisão

Ainda não temos estratégia.

O que temos agora é melhor do que as rodadas anteriores: uma base prospectiva com book/depth fresco,
mark e spot-perp, pronta para scouts. Isso encerra a dependência exclusiva de `aggTrades` e permite
testar sinais de microestrutura reais.

Mas a evidência desta rodada não autoriza operação:

- nenhuma ordem foi enviada;
- não há autorização para Demo, Testnet ou LIVE;
- a captura de 18 minutos é curta para aprovação estatística;
- todos os alvos 12/24 bps em 30s ficaram falsos;
- o resultado aponta que os custos/alvos usados nas rodadas anteriores eram grandes demais para esse
  regime intradiário curto.

Próximo caminho válido: rodar sensibilidade de alvos/custos sobre as linhas com book, por exemplo
1/2/3/5/8/12 bps, e depois repetir em capturas mais longas e em regimes de maior volatilidade. Só faz
sentido procurar modelo depois de sabermos qual alvo é fisicamente alcançável com spread, slippage e
latência realistas.

## Artefatos

- Script: `backend/scripts/research/run_book_wal_scout.py`
- Teste: `backend/tests/unit/test_book_wal_scout.py`
- Relatório v9 ignorado pelo Git: `backend/data/microstructure_v1/reports/book-wal-scout-v9/report.json`
  (`f923c59db4db7e724f9179c0b57bd77e4a88fb27c1003b05864172a67447b02a`)
- Rows v9 ignorado pelo Git: `backend/data/microstructure_v1/reports/book-wal-scout-v9/research_rows.parquet`
  (`5602396e80bfdfe9956a495e618d1ca16d3db9e4595f64e225aa483b35af96a7`)
- Relatório Fast16 ignorado pelo Git: `backend/data/microstructure_v1/reports/book-wal-scout-fast16-18m/report.json`
  (`4f6685790183dc9ead1a359de696d997570659c2ce829fa51af0d73b2eda4ab0`)
- Rows Fast16 ignorado pelo Git: `backend/data/microstructure_v1/reports/book-wal-scout-fast16-18m/research_rows.parquet`
  (`76ff5bcb638d15822144f87b5aa25c77426621c3cd90b6b859b299360fe7c95d`)
- Audit status Fast16 ignorado pelo Git:
  `backend/data/microstructure_v1/prospective-wal-fast16-18m-audits/evidence-gate-status.json`
  (`84cb80978f6134eaf2b0fb9ae60245bc145b47f69163f63df37fe3d0e58a4472`)
