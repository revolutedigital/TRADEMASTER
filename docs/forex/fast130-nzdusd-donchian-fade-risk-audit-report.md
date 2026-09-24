# Fast130 — NZDUSD Donchian fade risk audit

Fast130 pegou o único candidato aprovado no Fast129 e testou gestão de risco
com stop/target/trailing. Esta é a primeira pista da rodada atual que sobreviveu
também com hard stop.

Status:

`FAST130_NZDUSD_DONCHIAN_FADE_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION`

## Hipótese congelada

Entrada:

- par: `NZDUSD`;
- sinal: Donchian fade em D1;
- canal: `100` dias;
- direção: contra o rompimento;
- entrada: M1 posterior ao fechamento D1;
- hold máximo: `20` dias.

Risco:

- base de risco: ATR(14) diário, deslocado um D1 para trás;
- stops testados: `1.0`, `1.5`, `2.0` ATR;
- gestões testadas:
  - `stop_only`;
  - `fixed_1r`;
  - `fixed_2r`;
  - `trail_1r_1r`.

## Resultado

Números:

- sinais de entrada: `79`;
- políticas de risco: `12`;
- trades simulados: `780`;
- finalistas elegíveis na seleção: `10`;
- candidatos aprovados no audit: `9`.

Melhor por ranking de seleção/audit:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`

Seleção:

- trades: `44`;
- média base: `+45.2339` pips/trade;
- média stress: `+40.9656` pips/trade;
- LCB stress seleção: `+2.8862`;
- anos positivos: `4/5`;
- risco médio: `132.9373` pips.

Audit:

- trades: `21`;
- média base: `+60.3970` pips/trade;
- média stress: `+58.9756` pips/trade;
- LCB stress Bonferroni: `+13.1163`;
- anos positivos: `3/3`;
- `pilot_gate=true`.

Outros aprovados relevantes:

- `stop2_stop_only`: audit stress `+60.5504`, LCB `+12.1980`;
- `stop2_fixed_1r`: audit stress `+67.7507`, LCB `+32.5270`;
- `stop1.5_fixed_1r`: audit stress `+65.8002`, LCB `+44.1316`;
- `stop1_fixed_2r`: audit stress `+45.3261`, LCB `+1.7963`.

## Interpretação

Temos uma hipótese estatística de verdade, mas ainda não uma estratégia
operável. O candidato é promissor porque:

- sobreviveu seleção e audit;
- passou com hard stop;
- passou em cenário stress;
- teve anos positivos em `3/3` no audit;
- o LCB audit ficou positivo mesmo com Bonferroni.

Ainda assim, não está aprovado porque:

- foi descoberto por busca histórica;
- a amostra é pequena (`44` seleção, `21` audit);
- precisa de observação prospectiva fresca;
- precisa de revisão de execução real, tamanho de posição e gaps de fim de
  semana antes de qualquer Testnet.

## Segurança

Fast130 é pesquisa offline:

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
  `backend/scripts/research/run_fast130_nzdusd_donchian_fade_risk_audit.py`;
- teste:
  `backend/tests/unit/test_fast130_nzdusd_donchian_fade_risk_audit.py`;
- report JSON:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/report.json`;
- sinais:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/entry-signals.parquet`;
- trades:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/risk-audit-trades.parquet`;
- seleção:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/audited-finalists.csv`;
- qualidade:
  `backend/data/lab_fast130/nzdusd-donchian-fade-risk-audit/pair-quality.json`.

Hashes:

- report:
  `b6c473330d22aaf8f1f2a01671ade72bb09a3664cb817169acd9fa7a919007e3`;
- sinais:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- trades:
  `9b5049138cd4ac3fe27099a4dfe0a5890487ebd6521c5a530dcf7e7243438ce6`;
- seleção:
  `2ca0b42fea20428fa5ccd0b8c14ae1fd96b3aa11bb53abc2e14ee655a6962c0c`;
- audit:
  `816ce331c515205d7ab2bf3af90103956f3e45ed1ea5cf4be06f04316d180b5b`;
- pair quality:
  `2f1596c51b37aa93a5502b40ee6b9c16dfc24827c96b930162c6649897a14a5d`.

## Próximo passo

Pré-registrar a observação prospectiva do melhor candidato. O candidato
preferencial a congelar é:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`

Até haver dados prospectivos pós-pré-registro, o status correto continua sendo:
pesquisa candidata, não estratégia aprovada.
