# Rodada 28 — shadow prospectivo novo da candidata fixa

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast28 de 30 minutos  
Resultado: a candidata continuou positiva em 1 bps de custo total na janela nova, mas ainda quebra em 2 bps.

## O que foi testado

Regra congelada, sem reotimização:

- lado: `SELL`;
- entrada base: `directed_spot_perp_basis_bps >= 5.2032340279142275`;
- gate: `spot_update_age_ms <= 140.48`;
- horizonte: 120s;
- alvo: 5 bps;
- stop inicial: 20 bps;
- replay: entrada/saída marketable top-of-book, spread embutido no gross.

Limites preservados:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`;
- nenhuma ordem enviada.

## Captura Fast28

WAL:

`backend/data/microstructure_v1/prospective-wal-fast28-30m`

Cobertura:

| Stream | Eventos | Primeiro evento | Último evento |
| --- | ---: | --- | --- |
| trade | 88.650 | 2026-09-23T11:15:26.683Z | 2026-09-23T11:45:25.373Z |
| depth | 17.638 | 2026-09-23T11:15:26.607Z | 2026-09-23T11:45:25.485Z |
| spot_trade | 65.856 | 2026-09-23T11:15:26.868Z | 2026-09-23T11:45:25.836Z |
| mark_price | 1.376 | 2026-09-23T11:15:25Z | 2026-09-23T11:45:25.006Z |

WAL scout:

- status: `READY_FOR_SCOUT_MODELING`;
- decisões: 330;
- rows: 660;
- report hash: `8aaab5f5e8db6a52b902954b109a5515890e296acd3591937390f50b22e41b53`.

## Janela shadow

Usei o wrapper:

`backend/scripts/research/run_book_fixed_gate_prospective_window.py`

Ele calcula uma janela segura a partir da própria WAL, preservando histórico de features e folga para settlement.

Janela efetiva:

- start: `2026-09-23T11:16:00Z`;
- último decision time: `2026-09-23T11:43:20Z`;
- end exclusivo: `2026-09-23T11:43:25Z`;
- decisões: 329;
- sinais shadow: 36.

Hashes:

- wrapper report: `9c595846528e7f2266ce45736c475abebb8b07dcf7f550134aa0bebc85cacf6c`;
- shadow policy report: `2c8bac03af4dcb66e1c2979c09fa0904dd11c22af0a8c5f6ce76518a9c82ada8`;
- shadow decisions JSONL: `29e73060c412e996a31a42105ae20721f9f0a971a918059029cce71dbf5d09fc`.

## Settlement executável

Resultado por latência:

| Latência | Modo | Trades exec. | Gross médio | Net 1 bps | Net 2 bps | Saídas | MDD bps |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0ms | independente | 36 | +1,6797 | +0,6797 | -0,3203 | 14 TARGET / 22 TIMEOUT | 35,5650 |
| 0ms | uma posição | 13 | +1,7220 | +0,7220 | -0,2780 | 6 TARGET / 7 TIMEOUT | 11,6180 |
| 100ms | independente | 36 | +1,1173 | +0,1173 | -0,8827 | 13 TARGET / 23 TIMEOUT | 50,8114 |
| 100ms | uma posição | 13 | +1,7220 | +0,7220 | -0,2780 | 6 TARGET / 7 TIMEOUT | 11,6180 |
| 250ms | independente | 36 | +1,1847 | +0,1847 | -0,8153 | 13 TARGET / 23 TIMEOUT | 48,3880 |
| 250ms | uma posição | 13 | +1,7220 | +0,7220 | -0,2780 | 6 TARGET / 7 TIMEOUT | 11,6180 |
| 500ms | independente | 36 | +1,2078 | +0,2078 | -0,7922 | 13 TARGET / 23 TIMEOUT | 45,6604 |
| 500ms | uma posição | 13 | +1,4610 | +0,4610 | -0,5390 | 6 TARGET / 7 TIMEOUT | 11,4892 |
| 1000ms | independente | 36 | +1,3110 | +0,3110 | -0,6890 | 14 TARGET / 22 TIMEOUT | 45,3912 |
| 1000ms | uma posição | 13 | +1,5519 | +0,5519 | -0,4481 | 6 TARGET / 7 TIMEOUT | 11,4892 |

Arquivos principais:

- 100ms settlement report: `backend/data/microstructure_v1/reports/book-fixed-gate-prospective-window-fast28-30m-100ms/settlement-100ms/report.json`;
- 100ms settlement report hash: `e411815e9734f9032f355aa9204666f60d665e6326ef988e06b23152daf8bd1c`;
- 100ms independent parquet hash: `b044a4aff06b00e5e456622e367ea6b1c807c937d4273f6758fb94a7cf5fdd6f`;
- 100ms one-position parquet hash: `1fc7deed4343a37fe728446b0b58d79f82aebd8aa04c8b603851310d0325fb92`.

## Leitura

Isso é uma melhora real no quadro da candidata:

- Fast24 já tinha sido positivo em 1 bps;
- Fast28, uma janela futura nova, também ficou positivo em 1 bps;
- a latência não destruiu o sinal em 1 bps;
- o modo uma posição ficou melhor e com drawdown menor que o modo independente.

Mas ainda não é estratégia aprovada:

- a 2 bps de custo total o edge segue negativo;
- uma posição teve só 13 trades executáveis na janela;
- ainda são duas janelas curtas, não vários dias de shadow;
- o status correto segue `prospective_shadow_stream_required_before_testnet`.

## Decisão

Temos uma candidata de pesquisa que merece shadow prospectivo contínuo.

Não temos autorização nem evidência suficiente para Testnet/LIVE.

Próximo passo técnico: repetir o mesmo pipeline em múltiplas janelas novas e agregar o resultado por dia/janela, mantendo a regra congelada e custo de 1 bps/2 bps separados.
