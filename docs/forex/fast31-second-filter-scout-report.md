# Rodada 31 — scout de segundo filtro para a candidata fixa

Data: 2026-09-23  
Base: settlements Fast24 + Fast28 + Fast29, latência 100ms  
Resultado: apareceu um segundo filtro promissor in-sample, mas o walk-forward de seleção ficou instável. Não há aprovação de estratégia.

## Por que esta rodada existe

A regra `spot_age_fixed_gate_v1` falhou como estratégia standalone na Rodada 30:

- modo independente positivo em 1 bps;
- modo uma posição por vez falhou em Fast29;
- 2 bps continuou negativo.

Então o próximo passo correto não é trocar stop/trailing para maquiar saída. É perguntar:

> quais sinais dentro da regra fixa parecem bons, e quais parecem ruins?

## Ferramenta adicionada

`backend/scripts/research/run_book_fixed_gate_second_filter_scout.py`

O script:

- lê settlement reports dry-run;
- valida `research_only=true`;
- valida `order_submission_allowed=false`;
- valida `execution_authorization=none`;
- valida `approved_strategy=false`;
- junta settlements com `shadow_decisions.jsonl` via `feature_vector_sha256`;
- testa filtros causais simples sobre features disponíveis no momento da decisão;
- reaplica o constraint de uma posição por vez depois do filtro;
- roda um walk-forward leave-one-window-out para checar instabilidade.
- aceita `--fixed-candidate` para avaliar uma hipótese congelada em janela futura sem reescolher filtro.

Features testadas:

- `book_update_age_ms`;
- `spread_bps`;
- `spot_update_age_ms`;
- `directed_spot_perp_basis_bps`.

## Fonte e hash

Report:

`backend/data/microstructure_v1/reports/book-fixed-gate-second-filter-scout-fast31-f24-f28-f29/report.json`

Hash:

`4b896ad5ead109e53dc72a1b78844a238e9f0a063c888953f38108bad2f80202`

Status:

`SECOND_FILTER_SCOUT_IN_SAMPLE_CANDIDATE_WALKFORWARD_UNSTABLE`

Motivos:

- `walkforward_passed_folds:1<3`;
- `future_holdout_required_before_strategy_approval`.

## Melhor candidato in-sample

Filtro adicional:

`directed_spot_perp_basis_bps <= 5.578475508410952`

Interpretação: não basta vender quando o basis spot-perp favorece SELL; basis extremo demais parece piorar o comportamento sequencial. A hipótese é que o sinal útil está numa divergência moderada, não no extremo.

Resultado in-sample com uma posição por vez:

| Janela | Sinais após filtro | Trades uma posição | Net 1 bps | Net 2 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | --- |
| Fast24 | 20 | 6 | +1,3279 | +0,3279 | 4 TARGET / 2 TIMEOUT |
| Fast28 | 23 | 11 | +0,5054 | -0,4946 | 4 TARGET / 7 TIMEOUT |
| Fast29 | 5 | 4 | +0,7893 | -0,2107 | 2 TARGET / 2 TIMEOUT |

Agregado:

- sinais selecionados: 48;
- trades uma posição: 21;
- net ponderado em 1 bps: +0,7945 bps;
- pior janela em 1 bps: +0,5054 bps;
- janelas positivas em 1 bps: 3/3;
- janelas positivas em 2 bps: 1/3.

## Walk-forward

O diagnóstico honesto é que a seleção automática ainda é instável:

| Holdout | Candidato escolhido nos folds de desenvolvimento | Resultado no holdout |
| --- | --- | --- |
| Fast24 | `directed_spot_perp_basis_bps <= 5.578475508410952` | passou, +1,3279 bps |
| Fast28 | `book_update_age_ms >= 50.0` | falhou, -0,9241 bps |
| Fast29 | `book_update_age_ms <= 14.4` | sem seleção executável |

Isso quer dizer:

- o filtro de basis moderado é interessante;
- mas o processo de escolha ainda pode estar overfitando;
- portanto ele precisa de uma janela futura congelada.

## Decisão

Não temos estratégia.

Temos uma hipótese melhor do que a anterior:

`SELL fixed gate + directed_spot_perp_basis_bps <= 5.578475508410952`

Mas ela foi escolhida olhando Fast24/Fast28/Fast29. O próximo passo obrigatório é aplicar esse filtro, congelado, numa nova janela futura. Se falhar, descartamos ou rebaixamos para feature. Se passar, repetimos em mais janelas.

Nada de Testnet/LIVE.
