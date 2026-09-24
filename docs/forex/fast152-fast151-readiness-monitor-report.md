# Fast152 — Fast151 readiness monitor

Fast152 é o monitor fail-closed da política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA`

## Resposta curta

O monitor automatiza o próximo passo seguro: rerodar a observação Fast151,
confirmar se já há M1 posterior ao pré-registro e dizer se existe algo para
revisar.

Resultado atual:

- não há M1 posterior ao pré-registro Fast151;
- não há candidatos base pós-âncora;
- não há observações operacionais aceitas;
- não há ordem, ledger, Testnet, Demo ou LIVE.

## Resultado do monitor

| Métrica | Valor |
|---|---:|
| Status Fast152 | `FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA` |
| Status Fast151 rerodado | `FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET` |
| Raw M1 rows | `11265417` |
| Sinais reconstruídos | `344` |
| Trades materializados | `281` |
| Candidatos base pós-pré-registro | `0` |
| Observações Fast151 aceitas | `0` |
| Skips por posição | `0` |
| Skips por breaker | `0` |
| Gatilhos do breaker | `0` |

Fim dos dados por par:

| Par | Último M1 local |
|---|---|
| AUDUSD | `2026-09-18T20:59:00+00:00` |
| EURUSD | `2026-09-18T20:59:00+00:00` |
| GBPUSD | `2026-09-18T20:59:00+00:00` |
| NZDUSD | `2026-09-18T20:59:00+00:00` |

Pré-registro Fast151:

`2026-09-24T06:18:19-03:00`

Conclusão: o menor fim de dados por par ainda é anterior à âncora. O próximo
comando útil, quando houver dado novo, é rerodar:

```bash
backend/.venv/bin/python backend/scripts/research/run_fast152_fast151_readiness_monitor.py
```

## Interpretação

Fast152 não é mais uma tentativa de achar regra. Ele é o trilho de operação da
pesquisa:

- evita mexer na estratégia depois do pré-registro;
- evita contar histórico como forward;
- evita esquecer qual comando rerodar;
- evita promoção acidental antes de dados prospectivos.

Leitura honesta:

Temos uma candidata operacional histórica e pré-registrada. O monitor confirma
que, neste snapshot, ainda falta dado externo pós-âncora para começar a medir
forward.

## Segurança

Fast152 é monitoramento offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast152_fast151_readiness_monitor.py`;
- teste:
  `backend/tests/unit/test_fast152_fast151_readiness_monitor.py`;
- report JSON:
  `backend/data/lab_fast152/fast151-readiness-monitor/report.json`;
- Fast151 report rerodado:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/report.json`.

Hashes:

- Fast152 report:
  `39d0216b5d9d4d6ce4bf3e53a1f3b2d9bb479bb6d5ac7235742e818b4d0192cb`;
- Fast151 report rerodado:
  `602ee27425b8a4958f1a58240e552c379de3700124b9af28deb0532e77c59701`;
- Fast152 runner:
  `9b08e4d9c02d2e298fc46b19d76f5b72c227c387f936cfcc3035661e5c5b9fb6`;
- Fast152 test:
  `00ad092530ba5b1a022ea0303fce498e959d511f6f66c51a1cb3c951d426d6cb`.

## Próximo passo

Quando a fonte tiver M1 posterior a `2026-09-24T06:18:19-03:00`, rerodar o
Fast152. O status esperado passa para uma das opções:

- `FAST152_DATA_FRESH_NO_BASE_CANDIDATE_YET`: há dado novo, mas sem setup;
- `FAST152_DATA_FRESH_BUT_NO_ACCEPTED_OPERATIONAL_TRADE_YET`: há candidato,
  mas os controles operacionais ainda não geraram observação aceita;
- `FAST152_FAST151_OBSERVATION_READY_FOR_REVIEW`: existe trade aceito para
  revisar nos gates Fast151.
