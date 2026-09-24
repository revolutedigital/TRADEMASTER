# Fast175 — regime gate audit sobre o clue Fast174

Fast175 pegou o único clue do Fast174 que passou treino e validação:

`ETHUSDT`, horizonte `900s`, top fraction `0.02`.

A pergunta foi:

> Dá para adicionar um gate causal simples, escolhido sem olhar a extensão, que
> filtre o regime ruim e transforme o clue em algo robusto?

Status:

`FAST175_POSTHOC_REGIME_GATES_FOUND_BUT_PRESELECTED_GATE_FAILED`

## Resposta curta

Ainda não temos estratégia.

O gate honesto, escolhido só pelo treino, falhou. Mas apareceram `3` gates
pós-hoc que passariam treino, validação e extensão. Esses gates são pistas
úteis, não aprovação.

## Protocolo

Fonte:

- runner base: `backend/scripts/research/run_fast174_crypto_cross_asset_leadlag_smoke.py`;
- clue base: `ETHUSDT`, `900s`, top `0.02`;
- custo explícito: `2.0` bps.

Blocos:

- treino: `fast50`, `fast51`, `fast52`;
- validação: `fast53`;
- extensão/late holdout: `fast56`.

Gate family:

- gates simples de 1 feature;
- operadores: `<=`, `>=`;
- thresholds: quantis de treino `0.20`, `0.33`, `0.50`, `0.67`, `0.80`;
- features: retornos, volatilidade, flow, quote volume, hora, prediction e edge.

Critério para um gate entrar no audit:

- escolhido a partir do treino;
- mínimo `8` trades no treino;
- treino líquido positivo;
- pelo menos `2` blocos de treino representados.

Preseleção honesta:

- escolher sem olhar validação/extensão;
- ordenar por maior `min_train_block_net_bps`;
- depois por maior net total de treino;
- exigir todos os `3` blocos de treino representados.

## Resultado base sem gate

| Amostra | Trades | Net total | Média | Win rate |
|---|---:|---:|---:|---:|
| Treino | `34` | `+893.4026` | `+26.2765` | `61.76%` |
| Validação | `11` | `+29.1748` | `+2.6523` | `63.64%` |
| Extensão | `11` | `-272.3073` | `-24.7552` | `27.27%` |

## Gate pré-selecionado sem olhar holdout

Gate:

`hour_sin <= -0.1281436673`

Treino:

- trades: `23`;
- net: `+952.1546` bps;
- win rate: `78.26%`;
- blocos de treino: `3/3`;
- pior bloco de treino: `+0.9449` bps.

Validação:

- trades: `4`;
- net: `-96.8476` bps;
- win rate: `25.00%`.

Extensão:

- trades: `3`;
- net: `-102.6405` bps;
- win rate: `33.33%`.

Conclusão: o gate que seria escolhido de forma limpa pelo treino falhou fora do
treino. Portanto não pode virar candidato.

## Gates pós-hoc encontrados

O audit completo avaliou `391` gates de treino. Destes:

- `209` passaram validação;
- `3` passaram validação e extensão;
- `2` dos `3` também tinham todos os blocos de treino representados.

Top gates pós-hoc:

| Gate | Treino net | Validação net | Extensão net | Observação |
|---|---:|---:|---:|---|
| `ETHUSDT_qv_30m >= 586337978.3185` | `+508.9846` | `+66.2637` | `+30.6811` | Não tinha `fast51` no treino |
| `SOLUSDT_flow_1m >= 0.0319662448` | `+227.3519` | `+98.5681` | `+30.6652` | 3 blocos, mas `fast51` negativo |
| `ETHUSDT_ret_15m_bps >= -4.9038592392` | `+160.6781` | `+37.7757` | `+25.3853` | 3 blocos, mas `fast51` negativo |

Esses gates importam porque apontam para um padrão:

- alta liquidez/quote volume em ETH;
- flow positivo em SOL;
- ETH não muito fraco no retorno de 15 minutos.

Mas eles são pós-hoc. A extensão já foi olhada nesta rodada, então o próximo
uso honesto deles seria pré-registro prospectivo, não promoção.

## Interpretação

Fast175 melhora nossa visão:

1. O clue cross-asset do Fast174 não morreu por ausência total de sinal.
2. Gates de regime conseguem transformar o resultado histórico em algo positivo.
3. Mas a seleção honesta ex-ante ainda escolhe um gate ruim.
4. Portanto o problema agora é seleção de regime, não só entrada/target/trailing.

Decisão:

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O caminho promissor, se continuarmos nessa linha, é pré-registrar um dos gates
diagnósticos — ou uma regra simples derivada deles — e observar somente dados
novos. O melhor candidato diagnóstico bruto foi:

`ETHUSDT_qv_30m >= 586337978.3185`

Mas por ser escolhido depois de ver extensão, ele exige novo pré-registro
prospectivo antes de qualquer interpretação operacional.

## Segurança

Fast175 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado.

## Artefatos

- runner:
  `backend/scripts/research/run_fast175_fast174_regime_gate_audit.py`;
- teste:
  `backend/tests/unit/test_fast175_fast174_regime_gate_audit.py`;
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast175-fast174-regime-gate-audit/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast175-fast174-regime-gate-audit/summary.json`;
- gate evaluations:
  `backend/data/microstructure_cross_asset_v1/reports/fast175-fast174-regime-gate-audit/gate-evaluations.json`.

Hashes:

- report:
  `424b5f007874d52606d5a146fd9fbf3249ac985a9349e4e660def3f522204ab4`;
- summary:
  `5a7dbcb6eae8e6dade998caba453b66778ff6c2d8323f699368eb1bda95eafae`;
- gate evaluations:
  `d21afd2e6520e574160465e047457c129b647346f794ef82c9d3108747091132`.
