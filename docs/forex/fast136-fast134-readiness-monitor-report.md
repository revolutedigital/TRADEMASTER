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
- fim do M1 local: `2026-08-31T23:59:00+00:00`;
- há M1 pós-pré-registro: `false`;
- sinais históricos reconstruídos: `79`;
- sinais pós-pré-registro: `0`;
- trades históricos materializados pela regra congelada: `65`;
- observações completas elegíveis pós-pré-registro: `0`.

## Snapshot Donchian do último D1 local

Último candle D1 disponível:

`2026-08-31T23:59:00+00:00`

| Campo | Valor |
|---|---:|
| close | `0.592145` |
| canal superior 100D | `0.599435` |
| canal inferior 100D | `0.562645` |
| distância até rompimento superior | `72.9000` pips |
| distância até rompimento inferior | `295.0000` pips |
| gatilho mais próximo | rompimento superior → fade short |
| sinal ativo agora | `false` |
| ATR diário | `40.1857` pips |
| risco se houvesse sinal | `80.3714` pips |

Interpretação: no último dado local, NZDUSD estava mais perto de romper para
cima. Se um fechamento D1 rompesse o canal superior, a regra congelada buscaria
um fade vendido no primeiro M1 seguinte. Mas esse candle é de agosto, antes do
pré-registro; não conta como evidência prospectiva.

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
  `8d6690ace3a02042793d4ee326689c03e4e89c4bac8a3742cfac36c86f50fffd`;
- sinais pós-pré-registro:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.

## Próxima ação

Importar ou capturar M1 de NZDUSD posterior a
`2026-09-24T04:04:40-03:00` e rerodar:

1. Fast134, para avaliar observações prospectivas completas da regra congelada;
2. Fast136, para monitorar se já há sinal/entrada/resultado materializado.

Até lá, a resposta operacional continua: não aprovar estratégia, não Demo, não
Testnet, não LIVE.
