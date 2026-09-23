# Fast82 — primeira captura pós-pré-registro Fast80

Resposta objetiva: ainda não temos estratégia.

O Fast82 coletou o primeiro WAL local posterior ao pré-registro Fast80 e rodou
a cadeia congelada Fast60 → Fast61 → Fast80. O resultado não foi negativo por
custo ou trailing; foi ausência de entrada top-p no modelo congelado durante a
janela capturada.

## Safety boundary

Mantido em todo o fluxo:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Captura

Comando executado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/record_microstructure.py \
  --root backend/data/microstructure_v1/prospective-wal-fast82-30m \
  --symbol BTCUSDT \
  --duration-seconds 1800 \
  --include-spot-trades \
  --audit-status-path backend/data/microstructure_v1/prospective-wal-fast82-30m-audits/evidence-gate-status.json \
  --audit-rolling-days 1 \
  --audit-interval-seconds 60
```

O processo terminou com exit code `0`. No fim da captura houve reconnect por
timeout de ping em `spot_trade` e no stream de microestrutura, mas os arquivos
foram fechados e lidos com sucesso.

| Stream | Linhas | Primeiro evento | Último evento | SHA-256 |
|---|---:|---|---|---|
| futures trade | 53.851 | `2026-09-23T18:58:42.084000Z` | `2026-09-23T19:27:57.972000Z` | `61833b867dda7a3bc4e0dd961815f032dc631c41059b7a32a1ea7e274f17e32f` |
| depth | 17.217 | `2026-09-23T18:58:42.355652Z` | `2026-09-23T19:28:41.747367Z` | `e187ac095f55d7295376c1aea339921554ec78f6ca1e43c7d443fcb1680e2343` |
| mark price | 1.371 | `2026-09-23T18:58:41Z` | `2026-09-23T19:28:40.005000Z` | `5780a3c6b1e09d8cf8ab8f7d7030c3e5a600911ed6314504decfafb3e9316e2b` |
| spot trade | 51.610 | `2026-09-23T18:58:41.996000Z` | `2026-09-23T19:27:58.201000Z` | `f2ccc5eef5fbea0d52e96a6d5319b8690b196c0d257da60f23c10caaca4ba290` |

Esse WAL é posterior ao pré-registro Fast80:

- pré-registro Fast80: `2026-09-23T18:40:07Z`;
- primeiro evento Fast82: `2026-09-23T18:58:41Z`.

## Fast81/Fast80 update

Comando executado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast81_fast80_prospective_update.py \
  --wal-roots fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m \
  --output-root backend/data/microstructure_v1/reports/fast82-fast80-prospective-update
```

Resumo Fast81:

- status: `FAST81_FAST80_UPDATE_RESEARCH_ONLY`;
- WAL roots avaliados: `1`;
- WAL roots pós-pré-registro: `1`;
- último trade local usado: `2026-09-23T19:27:57.972000+00:00`;
- report SHA-256:
  `35680f5021215229f01886c689fb6db2152d90b38bb6299374f311fb3b8a7dcb`.

### Fast60

- status: `FAST60_NO_TOP_P_SIGNALS`;
- decisões: `339`;
- scored rows: `678`;
- raw top-p: `0`;
- gated signals: `0`;
- report SHA-256:
  `b7df5d29e833cda88940da6126b07c7cd4bf19ca5d184aac94384b4626e5ef2c`.

### Fast61

- status: `FAST61_NO_FAST60_GATED_SIGNALS`;
- signal count: `0`;
- complete outcomes: `0`;
- one-position outcomes: `0`;
- report SHA-256:
  `79b99af119745103ef4f839ae6645d50bd37bab3c11e22116ccec95561e1acb3`.

### Fast80

- status: `FAST80_OBSERVATION_NO_INPUT_RECORDS`;
- input records: `0`;
- post-preregistration records: `0`;
- rule matches: `0`;
- eligible observations: `0`;
- report SHA-256:
  `8bd276555750b0eb0c07e322a54c27c2a12067d09059f3862ecb16680b032fbb`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O Fast82 muda o diagnóstico: agora já existe WAL prospectivo pós-pré-registro,
mas o modelo top-p congelado não encontrou nenhuma entrada nessa janela de 30
minutos. A hipótese Fast80 continua aguardando mais exposição prospectiva; ela
não foi refutada por PnL nessa janela, porque não houve trade para liquidar.
