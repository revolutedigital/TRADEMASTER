# Fast162 — Fast151 management-variant audit

Fast162 auditou se a família da política operacional Fast151 melhora com
gestão alternativa de trade, especialmente trailing:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:*:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST162_FIXED_1R_REMAINS_PREFERRED_OVER_TRAILING`

## Resposta curta

Não. Dentro do grid histórico já materializado, trailing não melhora a candidata.

A política atual continua selecionada:

`stop2_fixed_1r`

O melhor trailing encontrado foi:

`stop1.5_trail_1r_1r`

Ele passa os gates, mas é mais fraco que `stop2_fixed_1r`:

| Variante | Pré LCB | Current LCB | Current swap BE | Current DD | Observação |
|---|---:|---:|---:|---:|---|
| `stop2_fixed_1r` | `+32.2878` | `+9.6335` | `0.9589` | `-616.6221` | selecionada |
| `stop1.5_trail_1r_1r` | `+26.7352` | `+4.0520` | `0.4655` | `-616.8421` | melhor trailing, mas pior |

## Principais variantes

| Variante | Gestão | Stop ATR | Pré trades | Pré LCB | Current trades | Current LCB | Current falhas |
|---|---|---:|---:|---:|---:|---:|---|
| `stop2_fixed_2r` | fixed_2r | `2.0` | `50` | `+34.9502` | `64` | `+11.7153` | `two_consecutive_negative_years` |
| `stop2_fixed_1r` | fixed_1r | `2.0` | `51` | `+32.2878` | `86` | `+9.6335` | — |
| `stop2_stop_only` | stop_only | `2.0` | `50` | `+31.5161` | `64` | `+2.9820` | `two_consecutive_negative_years` |
| `stop1.5_trail_1r_1r` | trail_1r_1r | `1.5` | `62` | `+26.7352` | `95` | `+4.0520` | — |
| `stop2_trail_1r_1r` | trail_1r_1r | `2.0` | `51` | `+27.0494` | `81` | `-3.4464` | `lcb_non_positive`, `two_consecutive_negative_years` |
| `stop1.5_fixed_1r` | fixed_1r | `1.5` | `72` | `+18.1083` | `95` | `+3.9020` | — |

## Interpretação

Esse audit responde diretamente a hipótese “trailar o stop deixa melhor?”:

- `stop2_fixed_2r` captura trades maiores e melhora current LCB, mas cria
  cluster de anos ruins no current;
- `stop2_stop_only` deixa correr mais, mas também cria anos ruins e perde muita
  margem de swap;
- `stop2_trail_1r_1r` reprova current LCB;
- `stop1.5_trail_1r_1r` é o melhor trailing, mas ainda perde para o benchmark em
  LCB current e tolerância a swap.

Leitura operacional:

- trailing não deve substituir a candidata atual;
- buscar “deixa correr mais” aumenta upside, mas piora estabilidade temporal;
- a política atual de alvo fixo 1R segue mais robusta para o objetivo de um bot
  menos frágil.

## Segurança

Fast162 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast162_fast151_management_variant_audit.py`;
- teste:
  `backend/tests/unit/test_fast162_fast151_management_variant_audit.py`;
- report JSON:
  `backend/data/lab_fast162/fast151-management-variant-audit/report.json`;
- métricas:
  `backend/data/lab_fast162/fast151-management-variant-audit/management-variant-metrics.csv`;
- resumo:
  `backend/data/lab_fast162/fast151-management-variant-audit/management-variant-summary.json`.

Hashes:

- report:
  `90eb02792193d4c0522cba4603ec1b69bc6e725178684d36b32e83e872268b7a`;
- runner:
  `bd3eeaf67bcdd8d6d880a50e5b8d36432bb1b783000b111d80325ac6d3ef33f9`;
- teste:
  `98229f0a02d5c2c22848b5c3f98fbb208ab6f1dcfe9637527f0472eea3ba27e7`;
- métricas:
  `91bd44a90b046d6de97dd04f5105d3862229b06eb458193c168320b70e26c2b6`;
- resumo:
  `d7b799cb934282f8caa560e60b552403ba048fe7bcc2d21a3bd5cc327fc00b94`.
