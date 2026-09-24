# Fast117 — inventário de dados e decisão do próximo ramo

Resposta objetiva: ainda não temos estratégia. Depois de Fast114–Fast116, o
próximo caminho não deve ser mais tuning de top-p/gate nos mesmos WALs. O
inventário local mostra que há dois caminhos reais e diferentes:

1. microestrutura rica, mas só em `BTCUSDT`;
2. FX multi-par com M1 longo, mas sem book/depth.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Microestrutura local

Arquivos normalizados/baixados:

- raw aggTrades: `backend/data/microstructure_v1/raw/aggTrades`;
- símbolos encontrados: `BTCUSDT`;
- arquivos raw `aggTrades`: `92`;
- range local: `2026-06-23` a `2026-09-22`;
- WALs ricos de book/depth usados nos scouts: `16` raízes `prospective-wal-fast*`
  com capturas de `depth`, `trade`, `spot_trade` e `mark_price`.

Leitura:

- temos book/depth rico o suficiente para diagnósticos de microestrutura;
- mas localmente isso é um universo de um mercado só;
- “multi-mercado” em microestrutura não é um experimento local imediato sem
  baixar/coletar novos dados equivalentes de outros símbolos.

## FX local

Base M1 HistData/Dukascopy local:

| Par | Arquivos |
|---|---:|
| AUDUSD | `2` |
| EURGBP | `2` |
| EURJPY | `2` |
| EURUSD | `2` |
| GBPJPY | `2` |
| GBPUSD | `2` |
| NZDUSD | `2` |
| USDCAD | `2` |
| USDCHF | `2` |
| USDJPY | `3` |

Range principal:

- `2019-01` a `2026-08` para 10 pares;
- `USDJPY` tem adicional parcial `2026-09`, mas sem validação oracle H1 local.

Leitura:

- FX é o melhor caminho local para comparação multi-par e estabilidade temporal;
- o limite é que não temos livro/depth, só bid/ask M1;
- o ramo USDJPY Fast112/Fast113 já está pronto para observação prospectiva, mas
  ainda não tem dado pós-pré-registro.

## Decisão de ramo

Encerrar, por enquanto, o tuning local:

```text
BTCUSDT book/depth
large-move
target 10 bps
trail 15 bps
top fraction/gate
```

Motivos:

- Fast114: top-p fino não salvou;
- Fast115: gate focado pós-top-p falhou;
- Fast116: regime-first single-feature melhorou parcialmente, mas ainda perdeu;
- microestrutura local não tem outros mercados equivalentes para testar
  generalização sem nova coleta.

Próximos caminhos que realmente mudam o problema:

1. coletar/baixar book/depth equivalente de outros mercados para comparar
   microestrutura;
2. voltar para FX multi-par M1 e testar estratégias menos dependentes de book;
3. aguardar/baixar dado prospectivo pós-pré-registro para o USDJPY Fast112;
4. construir um modelo de regime multivariado separado, mas só vale se houver
   mais janelas, porque com 13 janelas o risco de overfit segue alto.

## Comando de inventário executado

```bash
./backend/.venv/bin/python - <<'PY'
from pathlib import Path
from collections import Counter

raw = Path('backend/data/microstructure_v1/raw/aggTrades')
files = sorted(raw.glob('*-aggTrades-*.zip'))
symbols = Counter(p.name.split('-aggTrades-')[0] for p in files)
dates = sorted(p.name.split('-aggTrades-')[1].replace('.zip', '') for p in files)
print('aggtrade_symbols', dict(symbols))
print('aggtrade_dates', dates[0], dates[-1], len(dates))

wal_roots = sorted(
    p for p in Path('backend/data/microstructure_v1').glob('prospective-wal-fast*')
    if p.is_dir() and not p.name.endswith('-audits')
)
print('wal_roots', len(wal_roots))

fx = Path('backend/data/raw/fx_m1_hd')
fxfiles = sorted(fx.glob('*_M1_*.parquet'))
pairs = Counter(p.name.split('_M1_')[0] for p in fxfiles)
print('fx_pairs', dict(sorted(pairs.items())))
PY
```

Resultado relevante:

- `aggtrade_symbols {'BTCUSDT': 92}`;
- `aggtrade_dates 2026-06-23 2026-09-22 92`;
- `wal_roots 16`;
- `fx_pairs`: 10 pares, com `USDJPY` contendo 3 arquivos por causa de setembro
  parcial.
