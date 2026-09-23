# Fast88 — runner de observação directional-flow

Resposta objetiva: ainda não temos estratégia. Temos um runner testado para
observar prospectivamente a hipótese Fast88 sem usar dado antigo como aprovação.

Fast88 continua sendo apenas observação `research_only`.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## O que foi implementado

Runner:

`backend/scripts/research/run_fast88_directional_flow_observation.py`

Ele aplica a regra pré-registrada:

- `would_enter == true`;
- `realized_vol_30s_bps >= 15.0`;
- `directed_flow_imbalance_5s > 0.0`;
- conflito BUY/SELL no mesmo `label + decision_time_ms` resolvido por maior
  `probability`;
- settlement offline com política `micro_8_3`;
- uma posição por vez;
- gates de falha/candidato do pré-registro Fast88.

O runner falha fechado se as features obrigatórias faltarem. Ele não envia
ordem, não grava ledger e não libera Testnet/LIVE.

## Validação local

Teste unitário:

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast88_directional_flow_observation.py -q --no-cov
```

Resultado: `8 passed`.

Lint:

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast88_directional_flow_observation.py \
  backend/tests/unit/test_fast88_directional_flow_observation.py
```

Resultado: `All checks passed!`.

## Smoke contra Fast82/Fast83

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast88_directional_flow_observation.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast82-fast83/all_scored_decisions.jsonl \
  --wal-roots fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m \
              fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m \
  --output-root backend/data/microstructure_v1/reports/fast88-directional-flow-observation-smoke-fast82-fast83
```

Resultado:

- status: `FAST88_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET`;
- input records: `1.364`;
- post-preregistration records: `0`;
- candidate universe: `0`;
- rule matches: `0`;
- eligible observations: `0`;
- testnet/live: `false`.

Esse resultado é esperado e importante: Fast82/Fast83 são anteriores ao
pré-registro Fast88 (`2026-09-23T20:18:35Z`), então não podem confirmar a
hipótese. O runner recusou promover dado antigo.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O que existe agora:

1. Fast80 direto foi rejeitado;
2. Fast87 encontrou uma pista positiva pequena e não robusta;
3. Fast88 congelou essa pista como hipótese prospectiva;
4. o runner Fast88 está pronto para medir janelas novas;
5. ainda falta resultado prospectivo suficiente.

## Arquivos e hashes

- Fast88 runner:
  `73ebab306f2ca0e06eadb7a71f856dfa38419d252c44f08d4a4828d2d7de55eb`;
- Fast88 tests:
  `9ad54488a8eaa24b71cf71a457bed5a47f1c3943cc4f7a5dfe3e3b559900c598`;
- smoke report:
  `0493cf0111f4df7157c9f9d1b2dff2199742eed81babadc812591004b8eb10a9`.

## Próximo passo

Scorear uma janela WAL posterior ao pré-registro Fast88 e rodar este runner.
Se aparecer trade, a decisão continua pelo gate pré-registrado:

- falha precoce com `20` trades e stress total `<= -20` bps;
- falha por bloco com `10` trades e stress total `<= -10` bps;
- candidato apenas com `100` trades, `10` dias, stress médio `> +1.5`
  bps/trade, expected médio `> +2.5` bps/trade e lower bound de stress `> 0`.
