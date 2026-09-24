# Fast121 — scout de rompimento pós-compressão em FX H1/H4

Resposta objetiva: ainda não temos estratégia. Este ramo testou trades maiores
que os ramos intraday anteriores: romper range H1/H4 apenas depois de compressão
de volatilidade, com stop/alvo maiores e execução no M1 bid/ask. O gate formal
falhou, mas apareceu a melhor pista até agora em USDJPY H1.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## O que foi testado

Runner:

`backend/scripts/research/run_fast121_fx_compression_breakout_scout.py`

Base:

- `backend/data/raw/fx_m1_hd`;
- 10 pares: AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD,
  USDCHF e USDJPY;
- janela usada: `2019-01-01` até `2026-09-01`;
- setembro/2026 parcial e não-validado ficou fora.

Hipótese:

- agregar M1 para H1 e H4;
- calcular canal causal de 20 ou 55 barras;
- calcular compressão causal por range médio e quantil histórico;
- entrar comprado quando o fechamento rompe o topo do canal sob compressão;
- entrar vendido quando rompe o fundo;
- executar a entrada no próximo M1;
- stop = ATR agregado × `1.0` ou `1.5`;
- saídas: fixed 2R, fixed 3R ou trailing após 1R com distância 1R;
- holds máximos:
  - H1: 12h e 24h;
  - H4: 24h e 72h.

Total:

- `96` políticas por par;
- `960` candidatos par-política.

## Resultado

Status:

`FAST121_FX_COMPRESSION_BREAKOUT_FAILED_AUDIT`

Resumo:

| Métrica | Valor |
|---|---:|
| Candidatos par-política | `960` |
| Candidatos com trades na seleção | `864` |
| Trade rows simuladas | `183192` |
| Finalistas elegíveis | `50` |
| Candidatos aprovados | `0` |
| Estratégia aprovada | `false` |

O gate falhou porque nenhum finalista teve lower bound positivo na auditoria
2024–2026 após correção do conjunto de finalistas.

## Melhor pista: USDJPY H1 compressão + fixed 3R

A melhor linha com amostra decente foi:

```text
USDJPY
timeframe: H1
channel_bars: 20
compression_quantile: 0.40
stop_atr_multiple: 1.5
hold: 720 min
management: fixed_3r
```

Ela não passou o gate, mas é diferente dos ramos anteriores porque ficou positiva
em seleção e também positiva na auditoria:

| Amostra | Trades | Base médio | Stress médio | Lower stress | Anos positivos |
|---|---:|---:|---:|---:|---:|
| Seleção 2019–2023 | `524` | `+2.7122` | `+1.3050` | não aplicado | `3/5` |
| Auditoria 2024–2026 | `430` | `+5.2184` | `+3.8716` | `-2.6272` | `2/3` |

Leitura:

- tem sinal líquido depois de custo;
- tem amostra muito maior que as pistas de gap semanal;
- mas ainda há variância alta: o lower bound da auditoria fica negativo;
- por isso não pode virar bot.

Essa pista virou o Fast122: regra congelada, testada contra o histórico local
pré-2019.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast121 não encontrou estratégia aprovada, mas encontrou um candidato de pesquisa
mais interessante que os ramos anteriores. O caminho correto é tratá-lo como clue
histórico e auditar estabilidade temporal, não operar.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast121_fx_compression_breakout_scout.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast121_fx_compression_breakout_scout.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast121_fx_compression_breakout_scout.py \
  backend/tests/unit/test_fast121_fx_compression_breakout_scout.py
```

Resultado:

- unit tests: `5 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast121/fx-compression-breakout-scout/report.json`;
- report sha256:
  `bb0c010a94902d8c13ac9fcf4f5e5907b454d3a50f640f4a05195d0717133658`;
- trades:
  `backend/data/lab_fast121/fx-compression-breakout-scout/compression-breakout-trades.parquet`;
- trades sha256:
  `ac0064ec729fdef9b67f93da6310aa4aa3149cc9a06bf520e9e51f021ff2c504`;
- seleção:
  `backend/data/lab_fast121/fx-compression-breakout-scout/selection-metrics.csv`;
- seleção sha256:
  `dc66afabd3bfe343336e2e3f44942548439c2620018170514422f7ac3e206eb3`;
- auditoria:
  `backend/data/lab_fast121/fx-compression-breakout-scout/audited-finalists.csv`;
- auditoria sha256:
  `fa47761a3ff1067304c6ec427885d19cc2dbfcd38caf0c215631514e970f4881`;
- qualidade por par:
  `backend/data/lab_fast121/fx-compression-breakout-scout/pair-quality.json`;
- qualidade sha256:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.
