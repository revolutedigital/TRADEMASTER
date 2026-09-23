# Fast83 — segunda janela pós-pré-registro e leitura agregada Fast82+Fast83

Resposta objetiva: ainda não temos estratégia.

Fast83 adicionou mais uma janela prospectiva pós-pré-registro Fast80. O
resultado repetiu o Fast82: zero entrada top-p e zero ocorrência da regra
Fast80 vol+flow. A leitura agregada agora separa melhor os gargalos:

1. o modelo chega perto do threshold, mas ainda não cruza;
2. o regime Fast80 de volatilidade alta não apareceu em nenhuma linha scoreada.

## Safety boundary

Mantido em todos os comandos:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Captura Fast83

Comando executado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/record_microstructure.py \
  --root backend/data/microstructure_v1/prospective-wal-fast83-30m \
  --symbol BTCUSDT \
  --duration-seconds 1800 \
  --include-spot-trades \
  --audit-status-path backend/data/microstructure_v1/prospective-wal-fast83-30m-audits/evidence-gate-status.json \
  --audit-rolling-days 1 \
  --audit-interval-seconds 60
```

O processo terminou com exit code `0`. Houve reconnects no início da captura,
mas os quatro streams ficaram disponíveis e foram lidos no replay.

| Stream | Linhas | Primeiro evento | Último evento | SHA-256 |
|---|---:|---|---|---|
| futures trade | 60.408 | `2026-09-23T19:31:51.169000Z` | `2026-09-23T20:01:25.083000Z` | `356aa6bd65df0b45974842020712b2df6813131c5fffad65310f7ae1d7a070b5` |
| depth | 16.888 | `2026-09-23T19:31:51.834192Z` | `2026-09-23T20:01:25.294000Z` | `a40f0eafc1978210e7a31c8624cd5b74ec47670bc09db13af532818eee68350c` |
| mark price | 1.380 | `2026-09-23T19:31:25Z` | `2026-09-23T20:01:24Z` | `deee145c39dce4799acc423bfacae1968b9f691898d27592ce33e96cbbef1edd` |
| spot trade | 54.276 | `2026-09-23T19:32:14.848000Z` | `2026-09-23T20:01:25.579000Z` | `e20cb3f71a951095b33e88240de1ae4bc72222f41ce1f6c618373246449a2321` |

## Fast83 isolado

Fast81/Fast60/Fast61/Fast80:

- Fast60 status: `FAST60_NO_TOP_P_SIGNALS`;
- decisões: `343`;
- scored rows: `686`;
- raw top-p: `0`;
- gated signals: `0`;
- Fast61 status: `FAST61_NO_FAST60_GATED_SIGNALS`;
- Fast80 status: `FAST80_OBSERVATION_NO_INPUT_RECORDS`;
- Fast60 report SHA-256:
  `9a6264e8507f54dec39b2b435941c65c1de150cd52940266fb5962a74ec33268`;
- Fast84 diagnostic Fast83 report SHA-256:
  `bb3a69fbeb6e304ae21d4c46e8aa7597a8d560be044c91a06f43e4614bdfff76`.

Fast84 diagnóstico Fast83:

- scored rows: `686`;
- threshold: `0.90195311025804`;
- maior probabilidade: `0.8917379594024417`;
- distância até o threshold: `0.01021515085559832`;
- top-p entries: `0`;
- Fast80 rule matches: `0`;
- rows dentro de `0.02` do threshold: `15`;
- rows dentro de `0.05` do threshold: `171`.

## Agregado Fast82+Fast83

Comandos:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast81_fast80_prospective_update.py \
  --wal-roots fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m \
              fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m \
  --output-root backend/data/microstructure_v1/reports/fast83-fast80-prospective-update-combined-fast82-fast83
```

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast84_top_p_silence_diagnostic.py \
  --wal-roots fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m \
              fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m \
  --output-root backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast82-fast83
```

Resumo agregado:

- WAL roots pós-pré-registro: `2`;
- Fast81 status: `FAST81_FAST80_UPDATE_RESEARCH_ONLY`;
- Fast80 status: `FAST80_OBSERVATION_NO_INPUT_RECORDS`;
- scored rows: `1.364`;
- top-p entries: `0`;
- Fast80 rule matches: `0`;
- maior probabilidade: `0.89571932347643`;
- distância até o threshold: `0.006233786781610018`;
- rows dentro de `0.01` do threshold: `1`;
- rows dentro de `0.02` do threshold: `21`;
- rows dentro de `0.05` do threshold: `260`;
- rows dentro de `0.10` do threshold: `1.049`.

Distribuição agregada:

| Métrica | Probabilidade |
|---|---:|
| min | `0.7063506369856525` |
| p25 | `0.8038461137006275` |
| p50 | `0.8250161703361399` |
| p75 | `0.8463674750407849` |
| p90 | `0.8620593549123123` |
| p95 | `0.8713651785538776` |
| p99 | `0.8851049707311557` |
| max | `0.89571932347643` |
| mean | `0.8242558165236673` |

Hashes agregados:

- Fast81 combinado report:
  `33d6dc98c8d3a0b1e2354c047a6e3ee146b53fcbb23502a9b5bf4c9c007d9879`;
- Fast84 combinado report:
  `68887d11fb5e15a5194f22958957f58bf7e89d0e93b0d04c8b99d371fc1a5192`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O caminho Fast80 continua vivo como hipótese, mas as duas primeiras janelas
prospectivas mostram baixa frequência operacional:

- nenhum top-p em `1.364` rows;
- nenhuma ocorrência do regime vol+flow congelado;
- probabilidades relativamente próximas do threshold, sugerindo que uma busca
  futura pode explorar calibração, mas não neste pré-registro.

Próximo avanço útil: criar uma auditoria de frequência de regime/top-p em todas
as janelas WAL locais para decidir se vale continuar esperando Fast80 ou se
devemos trocar a família de sinal.
