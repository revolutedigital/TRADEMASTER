# Fast105a–Fast105b — segunda leitura prospectiva do veto fixo

Resposta objetiva: ainda não temos estratégia. A segunda janela foi positiva,
mas pequena e degradada por reconnects; o combinado A+B ficou praticamente zero
em custo esperado e negativo em stress.

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

## Captura Fast105b

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/record_microstructure.py \
  --root backend/data/microstructure_v1/prospective-wal-fast105b-30m \
  --symbol BTCUSDT \
  --duration-seconds 1800 \
  --include-spot-trades \
  --audit-status-path backend/data/microstructure_v1/prospective-wal-fast105b-30m-audits/evidence-gate-status.json \
  --audit-rolling-days 1 \
  --audit-interval-seconds 120
```

Observação importante: durante a janela Fast105b houve vários reconnects de
`microstructure_stream` e `spot_trade_stream` por timeout/handshake. O recorder
terminou com exit code `0`, mas a janela teve menos decisões elegíveis que
Fast105a (`160` contra `588`). Então o resultado positivo de Fast105b deve ser
lido como fraco.

## Comandos de avaliação arquivados

Fast105a isolado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast105_fixed_veto_prospective_observation.py \
  --validation-wal-roots fast105a=backend/data/microstructure_v1/prospective-wal-fast105a-30m \
  --output-root backend/data/microstructure_v1/reports/fast105a-prospective-observation
```

Fast105b isolado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast105_fixed_veto_prospective_observation.py \
  --validation-wal-roots fast105b=backend/data/microstructure_v1/prospective-wal-fast105b-30m \
  --output-root backend/data/microstructure_v1/reports/fast105b-prospective-observation
```

Fast105a + Fast105b:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast105_fixed_veto_prospective_observation.py \
  --validation-wal-roots fast105a=backend/data/microstructure_v1/prospective-wal-fast105a-30m,fast105b=backend/data/microstructure_v1/prospective-wal-fast105b-30m \
  --output-root backend/data/microstructure_v1/reports/fast105a-fast105b-prospective-observation
```

## Resultado por janela

| Janela | Decisões pós-pré-registro | Top 5% antes do veto | Após veto | Trades one-position | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|---:|---:|
| fast105a | `588` | `30` | `21` | `4` | `-10.019151404020977` | `-14.019151404020977` |
| fast105b | `160` | `8` | `7` | `2` | `+10.524744521861907` | `+8.524744521861907` |

Exit reasons:

- fast105a: `TIMEOUT: 4`;
- fast105b: `TIMEOUT: 2`.

## Resultado combinado A+B

Status:

`FAST105_INSUFFICIENT_PROSPECTIVE_EVIDENCE`

Resumo:

| Métrica | Valor |
|---|---:|
| Janelas elegíveis | `2` |
| Trades one-position | `6` |
| Sinais após veto | `28` |
| Gross total | `+6.505593117840928` bps |
| Net @ 1 bps | `+0.5055931178409283` bps |
| Net @ 2 bps | `-5.494406882159072` bps |
| Stress médio | `-0.9157344803598453` bps/trade |
| Fração de janelas positivas em stress | `0.5` |

Gates:

| Gate | Resultado |
|---|---|
| Mínimo `30` trades | Falhou (`6`) |
| Mínimo `3` janelas | Falhou (`2`) |
| Stress médio > `+1.5 bps/trade` | Falhou (`-0.9157`) |
| Expected médio > `+2.5 bps/trade` | Falhou (`+0.0843`) |
| Fração de janelas positivas em stress >= `75%` | Falhou (`50%`) |
| Falha precoce total | Não disparou |
| Falha precoce por janela | Não disparou |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Interpretação prática:

- Fast105a foi ruim;
- Fast105b foi bom, mas pequeno e capturado com reconnects;
- o agregado A+B perdeu dinheiro no custo de stress;
- o filtro `spot_return_1s_bps >= 0` ainda não demonstrou edge prospectivo.

O próximo teste, se continuar essa hipótese, precisa ser uma janela limpa
Fast105c. Se Fast105c vier fraca/negativa, eu encerraria esta linha sem gastar
mais coleta.

## Arquivos e hashes

Fast105a arquivado:

- report:
  `backend/data/microstructure_v1/reports/fast105a-prospective-observation/report.json`;
- report sha256:
  `7971da8887cd2d899632fa1ae203160af7b5c967636f0e2ac3e9518257dc9cdd`;
- one-position settlements sha256:
  `3aec0544631d7f446d76b374ca3e30a99ba585a5cd8f307dc80af6f8e528327d`.

Fast105b arquivado:

- report:
  `backend/data/microstructure_v1/reports/fast105b-prospective-observation/report.json`;
- report sha256:
  `a6271ec01a94af094dba44579bd459eb1d46e01a8f1ca740bb7aa9ff9fe0bbe9`;
- one-position settlements sha256:
  `187fd81f49acc0658a672bc6533ec780c008c80768623bbd9b8cef8f553e5f3c`.

Fast105a+b arquivado:

- report:
  `backend/data/microstructure_v1/reports/fast105a-fast105b-prospective-observation/report.json`;
- report sha256:
  `46c0bdbbb4ca8fe472ddeb9dd8bea1de6beaeae705dd07250d43e017ca0d7a24`;
- scored decisions sha256:
  `a5c8adf3bd8ebed8a5bdc51087d577fc7b244f51d52c6a5b8a5af7ad9f518d2d`;
- selected after fixed veto sha256:
  `6d8552830ff3f0a1c68afda71113ceb50388da3dbfa2758fd8235dfbb2d0a0e1`;
- one-position settlements sha256:
  `dc7f148c52bbe832950c2b13ace951dc03b83125288425135e251488d9585591`.
