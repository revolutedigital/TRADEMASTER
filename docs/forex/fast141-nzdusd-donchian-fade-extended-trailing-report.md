# Fast141 — NZDUSD Donchian fade extended trailing audit

Fast141 testou a pergunta prática que sobrou depois do Fast140: em vez de
microtrade curto, usar uma família de trades maiores e aplicar breakeven /
trailing melhora o candidato mais forte?

O candidato base é o mesmo do Fast134/Fast133:

`NZDUSD:d1_donchian_fade_ch100_hold20`

Status:

`FAST141_EXTENDED_TRAILING_DID_NOT_BEAT_FIXED_1R_BENCHMARK`

## Protocolo

Fonte:

- par: `NZDUSD`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- entrada: D1 Donchian fade, canal `100`, hold máximo `20` dias;
- risco: ATR(14) diário causal, deslocado um D1 antes do sinal;
- custos: `FUSION_ZERO` e `STRESS`, decisão primária por stress.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

Grade Fast141:

- stops: `1.0`, `1.5`, `2.0` ATR;
- gestões fixas: `fixed_1r`, `fixed_2r`, `stop_only`;
- trailing:
  - ativações: `0.5R`, `0.75R`, `1.0R`, `1.5R`;
  - trailing distances: `0.5R`, `0.75R`, `1.0R`;
- total: `45` políticas de risco.

## Resultado

| Métrica | Valor |
|---|---:|
| Sinais D1 reconstruídos | `79` |
| Linhas simuladas | `2,925` |
| Políticas de risco | `45` |
| Finalistas elegíveis | `35` |
| Políticas que passaram audit histórico | `27` |
| Melhor trailing bateu o benchmark `stop2_fixed_1r`? | `false` |

Comparação principal:

| Política | Seleção stress médio | Seleção LCB | Audit stress médio | Audit LCB | Audit anos positivos | Audit trades |
|---|---:|---:|---:|---:|---:|---:|
| `stop2_fixed_1r` | `+34.5342` | `+1.3261` | `+67.7507` | `+25.6904` | `3/3` | `21` |
| `stop2_trail_1r_0_75r` | `+25.8358` | `-7.1565` | `+65.1936` | `+10.3722` | `3/3` | `21` |
| `stop1.5_fixed_1r` | `+22.0184` | `-4.1391` | `+65.8002` | `+39.9259` | `3/3` | `21` |

Observações:

- o melhor audit médio continua sendo `stop2_fixed_1r`;
- o melhor trailing por audit médio, `stop2_trail_1r_0_75r`, ficou perto, mas
  abaixo do benchmark;
- o trailing passou em várias configurações, então a ideia de trailing não é
  absurda nesta família;
- porém, como troca de regra, ele não é melhor que o candidato já congelado;
- `stop1.5_fixed_1r` teve LCB de audit maior, mas seleção mais fraca e LCB de
  seleção negativo. Isso é uma pista diagnóstica, não motivo para trocar o
  pré-registro atual.

## Interpretação

Fast141 melhora a resposta estratégica:

- o problema principal não é simplesmente “falta trailing”;
- em trades maiores, o custo deixa de ser o vilão dominante;
- mas, para o melhor candidato atual, a gestão simples `fixed_1r` ainda é mais
  limpa que breakeven/trailing;
- portanto, o caminho correto continua sendo observar prospectivamente o
  `stop2_fixed_1r` já pré-registrado no Fast134, não trocar para uma regra nova
  depois de ver o histórico.

Isso não aprova estratégia. É evidência de que o candidato Fast134 continua
sendo a melhor pista operacionalmente simples.

## Segurança

Fast141 é pesquisa offline:

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
  `backend/scripts/research/run_fast141_nzdusd_donchian_fade_extended_trailing.py`;
- teste:
  `backend/tests/unit/test_fast141_nzdusd_donchian_fade_extended_trailing.py`;
- report JSON:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/report.json`;
- sinais:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/entry-signals.parquet`;
- trades:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/extended-trailing-trades.parquet`;
- seleção:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/audited-finalists.csv`;
- qualidade do par:
  `backend/data/lab_fast141/nzdusd-donchian-fade-extended-trailing/pair-quality.json`.

Hashes:

- report:
  `1863470f7f52eb89e20cd1a2f7726166da603002aa3bb1d7149f1a5d6e16491b`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades:
  `643eb780ea73b56cc694723fa872f6e78cace56f9f25f8f1a28232b740101867`;
- seleção:
  `c906e4e1fed1d6d3a8c62142fde89a9efdec44a44ac74cd9ccabafefdd7ebe76`;
- audit:
  `4795506dcfda8505d8ec25239b92e01618ea8cf62974c8960520feb71cd8b09c`;
- qualidade:
  `c2039204d066aa6a1966a3b9f4a5e252bfef22f6c03bdc8762317b3cd6ce7ffe`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Não trocar o Fast134 para trailing. O candidato `stop2_fixed_1r` continua sendo
o benchmark mais limpo. A próxima evidência que realmente muda a decisão é dado
M1 de NZDUSD posterior ao pré-registro do Fast134.
