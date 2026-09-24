# Fast105a — primeira janela prospectiva do veto fixo

Resposta objetiva: a primeira janela prospectiva do Fast105 foi negativa. Ainda
não há estratégia.

Fast105 foi pré-registrado e commitado antes da captura (`bb5251a`). Depois
disso foi gravada uma janela nova de mercado em:

`backend/data/microstructure_v1/prospective-wal-fast105a-30m`

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Comando de captura

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/record_microstructure.py \
  --root backend/data/microstructure_v1/prospective-wal-fast105a-30m \
  --symbol BTCUSDT \
  --duration-seconds 1800 \
  --include-spot-trades \
  --audit-status-path backend/data/microstructure_v1/prospective-wal-fast105a-30m-audits/evidence-gate-status.json \
  --audit-rolling-days 1 \
  --audit-interval-seconds 120
```

Streams gravados:

- `trade`;
- `depth`;
- `mark_price`;
- `spot_trade`.

Não houve envio de ordens; foi só captura pública de dados de mercado.

## Comando de avaliação

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast105_fixed_veto_prospective_observation.py \
  --validation-wal-roots fast105a=backend/data/microstructure_v1/prospective-wal-fast105a-30m
```

## Resultado

Status:

`FAST105_INSUFFICIENT_PROSPECTIVE_EVIDENCE`

Motivos:

- `not_approved_strategy`;
- `research_only`;
- `prospective_observation_only`;
- `testnet_requires_new_preregistration_and_explicit_authorization`;
- `gate_not_met:minimum_trades`;
- `gate_not_met:minimum_windows`;
- `keep_collecting_or_stop_if_timeout`.

Resumo da janela:

| Métrica | Valor |
|---|---:|
| Decisões pós-pré-registro | `588` |
| Selecionadas top 5% antes do veto | `30` |
| Selecionadas após `spot_return_1s_bps >= 0` | `21` |
| Trades one-position | `4` |
| Gross total | `-6.019151404020978` bps |
| Net @ 1 bps | `-10.019151404020977` bps |
| Net @ 2 bps | `-14.019151404020977` bps |
| Exit reasons | `TIMEOUT: 4` |

Gates:

| Gate | Resultado |
|---|---|
| Mínimo `30` trades | Falhou (`4`) |
| Mínimo `3` janelas | Falhou (`1`) |
| Stress médio > `+1.5 bps/trade` | Falhou (`-3.5047878510052444`) |
| Expected médio > `+2.5 bps/trade` | Falhou (`-2.5047878510052444`) |
| Fração de janelas positivas em stress >= `75%` | Falhou (`0%`) |
| Falha precoce total | Não disparou |
| Falha precoce por janela | Não disparou por pouco (`-14.019...`, limite `-15`) |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Interpretação prática:

- o primeiro teste prospectivo do filtro veio ruim;
- a amostra é pequena demais para encerrar estatisticamente;
- mas o sinal prospectivo inicial contradiz a melhora vista no Fast104;
- se as próximas janelas repetirem isso, o caminho deve ser encerrado rápido.

## Próximo passo

Coletar no máximo mais algumas janelas prospectivas Fast105, sem alterar regra.
Encerrar se qualquer gate de falha precoce disparar:

- após pelo menos `10` trades one-position, stress total <= `-20 bps`;
- qualquer janela com pelo menos `3` trades e stress total <= `-15 bps`.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast105-fixed-veto-prospective-observation/report.json`;
- report sha256:
  `87c173a9c412568488828b1ab3fb300aae95493594d4dcc5447d454f5af108f6`;
- scored decisions sha256:
  `35beaef6b2971c41255fba4b992b40225e355e754b9c11afb990e50e573d6b48`;
- selected after fixed veto sha256:
  `18101564f645c10851200c8b75ecd201b6f380fd2e1871820818364718ebf2c9`;
- one-position settlements sha256:
  `3aec0544631d7f446d76b374ca3e30a99ba585a5cd8f307dc80af6f8e528327d`.
