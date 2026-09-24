# Fast151 — operational observation run

Fast151 rodou a observação prospectiva da política operacional congelada:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET`

## Resposta curta

A candidata operacional foi congelada e o runner prospectivo existe, mas ainda
não há nenhum trade observável depois do pré-registro.

Motivo objetivo:

- pré-registro Fast151: `2026-09-24T06:18:19-03:00`;
- fim dos dados M1 locais em todos os pares: `2026-09-18T20:59:00+00:00`.

Ou seja: o banco local ainda termina antes da âncora prospectiva. Portanto,
qualquer “resultado” agora seria histórico e não pode contar como confirmação.

## Resultado do run

| Métrica | Valor |
|---|---:|
| Sinais reconstruídos | `344` |
| Trades materializados da política base | `281` |
| Candidatos base pós-pré-registro | `0` |
| Observações finais Fast151 | `0` |
| Skips por `max_per_pair_1` | `0` |
| Skips por breaker | `0` |
| Gatilhos do breaker | `0` |
| Raw M1 rows | `11265417` |

Fim dos dados por par:

| Par | Último M1 local |
|---|---|
| AUDUSD | `2026-09-18T20:59:00+00:00` |
| EURUSD | `2026-09-18T20:59:00+00:00` |
| GBPUSD | `2026-09-18T20:59:00+00:00` |
| NZDUSD | `2026-09-18T20:59:00+00:00` |

## Interpretação

Isso é um bom resultado de higiene, não de performance:

- a política está congelada;
- o runner valida o pré-registro;
- reconstrói sinais/trades com os dados atuais;
- aplica a janela prospectiva correta;
- aplica os controles operacionais na ordem certa;
- recusa confirmar qualquer coisa enquanto não houver dado posterior à âncora.

Leitura honesta:

Temos uma estratégia candidata operacional histórica e pré-registrada. Ainda
não temos confirmação prospectiva, porque o dataset local não alcança a data do
pré-registro.

## Segurança

Fast151 observation é pesquisa offline:

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
  `backend/scripts/research/run_fast151_usd_quoted_basket_operational_observation.py`;
- teste:
  `backend/tests/unit/test_fast151_usd_quoted_basket_operational_observation.py`;
- report JSON:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/report.json`;
- sinais:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/entry_signals.parquet`;
- trades materializados:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/materialized_trade_rows.parquet`;
- candidatos base pós-âncora:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/base-prospective-candidates.parquet`;
- observações JSONL:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/observations.jsonl`;
- observações parquet:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/observation-trade-rows.parquet`;
- skips posição:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/position-skipped.csv`;
- skips breaker:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/breaker-skipped.csv`;
- triggers breaker:
  `backend/data/lab_fast151/usd-quoted-basket-operational-observation/breaker-triggers.csv`.

Hashes:

- report:
  `b8e17e9e761e59079b7209cd137b9d1a593545230861c7b399e9ba4cff43b7a7`;
- runner:
  `3378c45f40bdb026dd2efaaa8e3f6a8e60adf285451b74789a20bed36a3e345f`;
- teste:
  `a8034a8eca97dfce594169269f6064f684ed746228025e49b5e4c295c5eec189`;
- pré-registro:
  `7a6ded6a5846d36faa609d2e2cdfd10d856a35f8b08bc8521c181443e3981856`;
- registry antes do evento de observação:
  `7e2be99e33ce6451b9bb33d6470966961ff4a33331d3cb7e641fc3612c15f625`;
- sinais:
  `1d3d8bb7fa92a7c4f462d1014b2662dc475a594845079527ee1b936aa6e0991c`;
- trades materializados:
  `2f9146cc6435e393c83d6235dd7ac0dfc6b2fe386b0ce9984d7838c6e9c394b0`;
- candidatos base pós-âncora:
  `c542498f9887d0729850ac5f3a6e8b670a111772602de7e59f610946331949f0`;
- observações JSONL:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observações parquet:
  `fe3949150810cef6eee186a082161817b89f66179655108466988c5a133ff303`;
- skips posição:
  `ae5e35f9e0cd1f399db7dfdde0c488175e3ec502bfb0bda897a3aeaedeef23bd`;
- skips breaker:
  `65441c313def440e1ae71024bca1de08769e1cf852c4a3214f4a966488ad7381`;
- triggers breaker:
  `81fd67ec37638a709fbe3e4188d9b538d5cb73cc8c437ca691c77f006c030551`.

## Próximo passo

Atualizar os dados M1 para período posterior a `2026-09-24T06:18:19-03:00` e
rerodar exatamente o mesmo runner Fast151. Se continuar sem trades, aguardar
novos candles; se aparecerem trades, avaliar apenas pelos gates já
pré-registrados.
