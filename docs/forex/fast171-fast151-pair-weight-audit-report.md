# Fast171 — Fast151 pair-weight audit

Fast171 auditou se a Fast151 melhora ao manter a mesma regra, mas reduzir ou
aumentar risco por par.

Escopo:

- mesmos 4 pares: `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- mesma entrada: `d1_donchian_fade_ch100_hold20`;
- mesmo risco: `stop2_fixed_1r`;
- mesmo lado: `SHORT`;
- mesmo overlay: `max_per_pair_1`;
- mesmo breaker: `dd800_cooldown180d`;
- pesos testados por par: `0.25`, `0.50`, `0.75`, `1.00`.

Status:

`FAST171_PAIR_WEIGHT_PASSES_CURRENT_BUT_NO_MATERIAL_SWAP_UPLIFT`

## Resposta curta

Não vale trocar a Fast151 equal-weight por ponderação de pares.

O melhor peso escolhido só pelo pré-2019 passa no current, mas piora exatamente
o blocker que queríamos resolver:

- baseline current swap BE: `0.9589`;
- peso escolhido no pré current swap BE: `0.5535`;
- uplift: `-0.4054`.

O melhor peso olhando os dois blocos melhora o current swap BE só de `0.9589`
para `0.9606`, diferença irrelevante, e ainda reduz LCB.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| políticas de peso avaliadas | `256` |
| passaram pré-2019 | `20` |
| passaram current 2019-2026 | `243` |
| passaram os dois blocos | `20` |

## Comparativo

| Política | Pesos | Pré LCB | Pré swap BE | Current LCB | Current swap BE | Decisão |
|---|---|---:|---:|---:|---:|---|
| baseline | todos `1.00` | `+32.2878` | `3.1040` | `+9.6335` | `0.9589` | manter |
| selecionada no pré | AUD `0.25`, demais `1.00` | `+29.7490` | `3.2442` | `+4.4142` | `0.5535` | rejeitar |
| melhor estrita diagnóstica | GBP `0.75`, demais `1.00` | `+27.5888` | `2.8190` | `+9.0542` | `0.9606` | ganho irrelevante |

## Interpretação

Essa auditoria fecha outra rota operacional: tentar resolver swap/drawdown só
com alocação por par.

O método foi conservador:

- partiu dos trades Fast149 já filtrados por `max_per_pair_1`;
- aplicou pesos no P/L por par;
- reaplicou o circuit breaker usando equity ponderada;
- selecionou a política usando somente o pré-2019;
- usou o current apenas como auditoria.

O resultado é claro:

- downweight de AUD melhora o pré, mas destrói margem no current;
- downweight de GBP quase não melhora swap current;
- nenhuma ponderação entrega melhoria material suficiente para substituir a
  Fast151 equal-weight.

## O que isso muda no caminho

Fast171 reduz o espaço de busca:

- não precisamos insistir em pair weighting simples como solução do swap;
- a Fast151 equal-weight segue sendo a versão mais limpa;
- o blocker real continua sendo medir swap da corretora e obter dados
  prospectivos pós-pré-registro.

## Segurança

Fast171 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast171_fast151_pair_weight_audit.py`;
- teste:
  `backend/tests/unit/test_fast171_fast151_pair_weight_audit.py`;
- report JSON:
  `backend/data/lab_fast171/fast151-pair-weight-audit/report.json`;
- métricas:
  `backend/data/lab_fast171/fast151-pair-weight-audit/pair-weight-metrics.csv`;
- resumo:
  `backend/data/lab_fast171/fast151-pair-weight-audit/pair-weight-summary.json`.

Hashes:

- report:
  `990e2ad5075603ef01770af558724b69a92c4e03c417458621eccd720d752af4`;
- runner:
  `d3884385e1d87dd21a9957e41c308a0af5a08f110c90d29802688bb84b5085bf`;
- teste:
  `62b6bbc5a6aa7c9977c18ec14325c67f3dcfe729d81e4d96834efe441042182c`;
- métricas:
  `678239639953f0f68fbcc8c315b6d3be643c90e90320e85dc644f8c7837347e3`;
- resumo:
  `d86bbb9478eadb1e5a2c346572097841c532574a3394ce5b7c5b6471eb124d19`.
