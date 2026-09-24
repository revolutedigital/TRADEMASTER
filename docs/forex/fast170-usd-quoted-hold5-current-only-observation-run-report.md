# Fast170 — USD-quoted `hold5` current-only observation run

Fast170 rodou o observador prospectivo da hipótese congelada:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold5:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST170_OBSERVATION_NO_ELIGIBLE_PROSPECTIVE_TRADES_YET`

## Resposta curta

O observador está criado e validado, mas ainda não há dados prospectivos
elegíveis.

O M1 local usado pelo run ainda termina em:

`2026-09-18T20:59:00+00:00`

Como o pré-registro Fast170 foi cravado em:

`2026-09-24T08:08:53-03:00`

nenhum trade posterior à âncora pode existir nesse banco ainda.

## Números do run

| Métrica | Valor |
|---|---:|
| sinais reconstruídos | `344` |
| trades materializados da regra | `217` |
| candidatos pós-pré-registro | `0` |
| observações elegíveis | `0` |
| pulados por posição | `0` |
| pulados por breaker | `0` |
| triggers de breaker | `0` |

## Interpretação

Fast170 não achou estratégia nova; ele colocou o melhor diagnóstico
current-only sob observação prospectiva limpa.

Isso é importante porque `ch100_hold5` ficou muito bom no current, mas falhou no
pré-2019. Sem esse pré-registro, qualquer uso futuro dele seria overfit. A
partir de agora, se novos dados M1 independentes chegarem, o runner consegue
avaliar a regra sem alterar parâmetros.

## Segurança

Fast170 é observação offline:

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

- pré-registro:
  `docs/forex/fast170-usd-quoted-hold5-current-only-prospective-preregistration.md`;
- registry:
  `docs/forex/fast170-registry.jsonl`;
- runner:
  `backend/scripts/research/run_fast170_usd_quoted_hold5_current_only_observation.py`;
- teste:
  `backend/tests/unit/test_fast170_usd_quoted_hold5_current_only_observation.py`;
- report JSON:
  `backend/data/lab_fast170/usd-quoted-hold5-current-only-observation/report.json`;
- observações:
  `backend/data/lab_fast170/usd-quoted-hold5-current-only-observation/observations.jsonl`;
- trades materializados:
  `backend/data/lab_fast170/usd-quoted-hold5-current-only-observation/materialized_trade_rows.parquet`.

Hashes:

- pré-registro:
  `46599cdf52c391147fe518ddccac8fa815cefae3642216d5b2023e5e6a66d025`;
- report:
  `3f441d5c878f633ef36af8a2a6f491e3b170d20e3eed6612c8785114084e6f80`;
- runner:
  `7b814bbcc4fdfceb0f0f6907bb00bfbb6fbd1756ba60e0dc0493fb877ae488d3`;
- teste:
  `3cda74bb01de4c13294912824399106ea7931de941df8ed4301997d295092ba4`;
- sinais:
  `1d3d8bb7fa92a7c4f462d1014b2662dc475a594845079527ee1b936aa6e0991c`;
- trades materializados:
  `e568fc8ecdf645adb404b903bbdb0f6681382a083fa64f71a5569cc9a5444391`;
- observações:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
