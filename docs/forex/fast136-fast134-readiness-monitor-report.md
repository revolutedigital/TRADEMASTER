# Fast136 — Fast134 readiness monitor

Fast136 criou um radar fail-closed para a candidata congelada do Fast134:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

Status:

`FAST136_WAITING_FOR_POST_PREREGISTRATION_M1_DATA`

## Resposta direta

Ainda não existe estratégia aprovada.

O que existe é uma candidata estatística pré-registrada. O monitor confirmou
que a base M1 local ainda termina antes do pré-registro do Fast134, então não
há como observar trade prospectivo novo nessa regra com os dados atuais.

## Evidência atual

- pré-registro Fast134: `2026-09-24T04:04:40-03:00`;
- início do M1 local: `2019-01-01T22:02:00+00:00`;
- fim do M1 local: `2026-09-18T20:59:00+00:00`;
- há M1 pós-pré-registro: `false`;
- sinais históricos reconstruídos: `79`;
- sinais pós-pré-registro: `0`;
- trades históricos materializados pela regra congelada: `65`;
- observações completas elegíveis pós-pré-registro: `0`.

Nota de dado: o mês `2026-09` de NZDUSD foi materializado localmente, mas o
arquivo disponível vai só até `2026-09-18T20:59:00+00:00`, ainda antes do
pré-registro. Como o oracle H1 local termina em agosto, setembro entra como
unverified/fail-closed no resumo de validação.

## Snapshot Donchian do último D1 local

Último candle D1 disponível:

`2026-09-18T23:59:00+00:00`

| Campo | Valor |
|---|---:|
| close | `0.572300` |
| canal superior 100D | `0.599435` |
| canal inferior 100D | `0.562645` |
| distância até rompimento superior | `271.3500` pips |
| distância até rompimento inferior | `96.5500` pips |
| gatilho mais próximo | rompimento inferior → fade long |
| sinal ativo agora | `false` |
| ATR diário | `45.9893` pips |
| risco se houvesse sinal | `91.9786` pips |

Interpretação: no último dado local, NZDUSD estava mais perto de romper para
baixo. Se um fechamento D1 rompesse o canal inferior, a regra congelada buscaria
um fade comprado no primeiro M1 seguinte. Mas esse candle é de `2026-09-18`,
antes do pré-registro; não conta como evidência prospectiva.

## Segurança

Fast136 é apenas monitoramento offline:

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
  `backend/scripts/research/run_fast136_fast134_readiness_monitor.py`;
- teste:
  `backend/tests/unit/test_fast136_fast134_readiness_monitor.py`;
- report JSON:
  `backend/data/lab_fast136/fast134-readiness-monitor/report.json`;
- sinais pós-pré-registro:
  `backend/data/lab_fast136/fast134-readiness-monitor/post-preregistration-signals.json`.

Hashes:

- report:
  `fda64b819bf26f6144f0de2c5748b1aeac4be6992d5785ff61f94e47065aa432`;
- sinais pós-pré-registro:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.

## Próxima ação

Importar ou capturar M1 de NZDUSD posterior a
`2026-09-24T04:04:40-03:00` e rerodar:

1. Fast134, para avaliar observações prospectivas completas da regra congelada;
2. Fast136, para monitorar se já há sinal/entrada/resultado materializado.

Até lá, a resposta operacional continua: não aprovar estratégia, não Demo, não
Testnet, não LIVE.
