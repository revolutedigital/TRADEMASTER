# Fast155 — Fast151 entry-filter audit

Fast155 testou uma hipótese importante do Igor:

> Dá para melhorar a entrada com um filtro tipo top-p/probabilidade usando
> sinais causais simples?

Status:

`FAST155_NO_ENTRY_FILTER_IMPROVES_FAST151_PRE_GATE`

## Resposta curta

Não com esses proxies simples.

Foram testados filtros causais em cima de:

- distância do rompimento em pips;
- distância do rompimento em R;
- ATR diário / regime de risco.

Aplicação correta:

- thresholds calibrados somente no pré-2019;
- filtro aplicado antes do `max_per_pair_1`;
- depois aplica `max_per_pair_1`;
- depois aplica `dd800_cooldown180d`;
- current 2019–2026 usado só como auditoria.

Resultado: o melhor filtro selecionado continua sendo `no_extra_filter`.

## Resultado resumido

| Filtro | Pré trades | Pré LCB | Pré passa? | Current trades | Current LCB | Current passa? |
|---|---:|---:|---|---:|---:|---|
| `no_extra_filter` | `51` | `+32.2878` | sim | `86` | `+9.6335` | sim |
| `breakout_distance_lte_pre_p50` | `36` | `-2.8699` | não | `55` | `+17.6665` | sim |
| `breakout_distance_lte_pre_p75` | `46` | `+7.4902` | não | `71` | `+8.2607` | sim |
| `breakout_distance_lte_pre_p90` | `52` | `+13.1592` | não | `77` | `+9.3587` | sim |
| `breakout_r_lte_pre_p50` | `35` | `+5.7561` | não | `53` | `+16.6574` | sim |
| `breakout_r_lte_pre_p75` | `47` | `+14.5569` | não | `70` | `+12.1944` | sim |
| `breakout_r_lte_pre_p90` | `51` | `+6.8602` | não | `76` | `+14.2614` | não |
| `atr_gte_pre_p25` | `38` | `+18.5355` | não | `46` | `+6.9252` | não |
| `atr_gte_pre_p50` | `33` | `+25.5233` | não | `32` | `+6.8969` | não |
| `atr_lte_pre_p75` | `38` | `+6.4760` | não | `69` | `+2.5109` | não |
| `atr_lte_pre_p90` | `44` | `+5.6657` | não | `72` | `-0.7557` | não |
| `atr_between_pre_p25_p75` | `26` | `-19.0174` | não | `31` | `-0.3476` | não |
| `risk_lte_pre_p75` | `38` | `+6.4760` | não | `69` | `+2.5109` | não |

## Interpretação

Vários filtros parecem sedutores porque melhoram o current. Exemplo:

- `breakout_distance_lte_pre_p50` deixa current com LCB `+17.6665`, melhor que
  o baseline.

Mas esse mesmo filtro falha no pré:

- só `36` trades;
- LCB `-2.8699`;
- anos ruins consecutivos.

Isso é exatamente a armadilha do “top-p” mal calibrado: parece ótimo no trecho
que estamos olhando agora, mas não sobrevive à seleção cronológica.

Leitura honesta:

Não devemos adicionar filtro de entrada simples ao Fast151. A estratégia
operacional congelada continua sendo:

`no_extra_filter + max_per_pair_1 + dd800_cooldown180d`

Se quisermos um modelo top-p de verdade depois, ele precisa nascer como outro
experimento pré-registrado, com treino/validação/teste e sem contaminar o
Fast151.

## Segurança

Fast155 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast155_fast151_entry_filter_audit.py`;
- teste:
  `backend/tests/unit/test_fast155_fast151_entry_filter_audit.py`;
- report JSON:
  `backend/data/lab_fast155/fast151-entry-filter-audit/report.json`;
- métricas:
  `backend/data/lab_fast155/fast151-entry-filter-audit/entry-filter-metrics.csv`;
- resumo:
  `backend/data/lab_fast155/fast151-entry-filter-audit/entry-filter-summary.json`.

Hashes:

- report:
  `120b6b08f53fd2632a44afae2586332700379bd42deed5d13c603200a7a4aaf4`;
- runner:
  `8a571671849025c710966195ba92c2dd5eee251023544dcd5b7900c3fe6445b2`;
- teste:
  `7dc735317c30f382ed6f93300c10e50ad0b2154de198d4fbc204c3e141325825`;
- métricas:
  `09b35c263102b423318cc462c9f2ff8a684e80ee63621ec127284b6a941b1b14`;
- resumo:
  `f6301a2820fab7749ba53233c73c8a453e87ffbd4db4e0e6b34eda8fea8df118`.
