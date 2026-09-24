# Fast120 — scout de força relativa cross-sectional em FX

Resposta objetiva: ainda não temos estratégia. O ramo testou uma hipótese mais
estrutural que par isolado: medir força/fraqueza relativa das moedas no conjunto
de pares e operar o par com maior divergência nas aberturas de sessão. Mesmo
assim, nenhuma política ficou positiva depois de custo de stress na seleção.

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

`backend/scripts/research/run_fast120_fx_currency_strength_scout.py`

Base:

- `backend/data/raw/fx_m1_hd`;
- 10 pares: AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD,
  USDCHF e USDJPY;
- janela usada: `2019-01-01` até `2026-09-01`;
- setembro/2026 parcial e não-validado ficou fora.

Hipótese:

1. calcular retorno logarítmico recente de cada par;
2. transformar retornos de pares em score de força por moeda:
   - base recebe `+retorno`;
   - quote recebe `-retorno`;
   - score final é média equal-weight das exposições;
3. calcular divergência de cada par negociável como `score(base) - score(quote)`;
4. nas sessões UTC `00:00`, `07:00` e `13:00`, escolher o par com maior
   divergência absoluta;
5. operar:
   - momentum: compra a moeda forte contra a fraca;
   - reversal: compra a fraca contra a forte;
6. entrada no próximo candle M1;
7. stop/target/trailing por risco causal baseado no range M1 anterior.

Grade:

- modos: momentum e reversal;
- lookbacks: 60, 240 e 1440 minutos;
- horas UTC: 00, 07 e 13;
- thresholds: 5, 10 e 20 bps;
- holds: 60 e 240 minutos;
- saídas: fixed 1R, fixed 2R, trailing após 1R com distância 0.75R.

Total: `324` políticas.

Custos:

- base: spread bid/ask observado + comissão Fusion Zero + slippage;
- stress: spread dobrado + slippage maior + comissão.

## Resultado

Status:

`FAST120_FX_CURRENCY_STRENGTH_FAILED_SELECTION`

Resumo:

| Métrica | Valor |
|---|---:|
| Políticas | `324` |
| Trade rows simuladas | `530154` |
| Políticas positivas em base | `7/324` |
| Políticas positivas em stress | `0/324` |
| Finalistas elegíveis | `0` |
| Candidatos aprovados | `0` |
| Estratégia aprovada | `false` |

Top da seleção por stress médio:

| Política | Trades | Base médio | Stress médio | Anos stress positivos | Top pair |
|---|---:|---:|---:|---:|---|
| `momentum_lb1440_h13_thr20_hold240_fixed_2r` | `1142` | `+0.3332` | `-1.2842` | `1/5` | NZDUSD |
| `momentum_lb1440_h13_thr10_hold240_fixed_2r` | `1145` | `+0.3056` | `-1.3092` | `1/5` | NZDUSD |
| `momentum_lb1440_h13_thr5_hold240_fixed_2r` | `1145` | `+0.3056` | `-1.3092` | `1/5` | NZDUSD |
| `momentum_lb1440_h13_thr20_hold240_trail_1r_0_75r` | `1142` | `+0.2458` | `-1.4682` | `1/5` | NZDUSD |
| `reversal_lb60_h07_thr20_hold240_fixed_2r` | `499` | `+0.0072` | `-1.6097` | `2/5` | AUDUSD |

Leitura:

- existe micro-sinal bruto em algumas configurações, porque `7/324` ficaram
  positivas em custo base;
- o stress mínimo já transforma todas em negativas;
- a melhor política perde `-1.2842` pips/trade em stress;
- trailing não salvou: as versões trailadas ficaram piores que fixed 2R nos
  melhores rankings;
- portanto o problema não é só escolher melhor a moeda forte/fraca; o edge é
  menor que custo/spread/slippage.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast120 mata a tese simples de força relativa cross-sectional intraday nesta
forma. O resultado é coerente com os ramos anteriores: há sinais pequenos no
bruto, mas o custo destrói o que sobra antes de virar uma estratégia defensável.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast120_fx_currency_strength_scout.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast120_fx_currency_strength_scout.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast120_fx_currency_strength_scout.py \
  backend/tests/unit/test_fast120_fx_currency_strength_scout.py
```

Resultado:

- unit tests: `7 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast120/fx-currency-strength-scout/report.json`;
- report sha256:
  `a6eb83e26ed116107d83e6e5bf73c416377fd3cbc026b51ec1a18cdcc711e23b`;
- trades:
  `backend/data/lab_fast120/fx-currency-strength-scout/currency-strength-trades.parquet`;
- trades sha256:
  `23380982fa4ca1eddec2621fca9d4132f5562aa633ab1e33188945da02848558`;
- seleção:
  `backend/data/lab_fast120/fx-currency-strength-scout/selection-metrics.csv`;
- seleção sha256:
  `6cf9ea0001be21f39eb274cdaed28fdad70a9ecb86309b1773ba02ffe42f85c3`;
- auditoria:
  `backend/data/lab_fast120/fx-currency-strength-scout/audited-finalists.csv`;
- auditoria sha256:
  `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`;
- qualidade por par:
  `backend/data/lab_fast120/fx-currency-strength-scout/pair-quality.json`;
- qualidade sha256:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.
