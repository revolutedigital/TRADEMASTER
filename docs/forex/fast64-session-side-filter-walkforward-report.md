# Fast64 — walk-forward de filtro sessão/lado/probabilidade

Depois de Fast61–Fast63, a pergunta ficou:

> existe um bolso causal simples — horário UTC, lado ou probabilidade mínima —
> que salve o top-p histórico?

Fast64 testa isso com leave-one-block-out nos blocos Fast43–Fast53. Em cada
holdout, o filtro é escolhido usando apenas os outros blocos; depois é congelado
e aplicado no bloco segurado.

## Escopo

- fonte: `frozen_holdout_trades.parquet` dos blocos Fast43–Fast53;
- mercado: BTCUSDT perpetual;
- filtros permitidos: side, sessão UTC, probabilidade mínima;
- filtros proibidos: qualquer informação pós-entrada, MFE/MAE, exit reason,
  gross/stress futuro;
- execução: one-position por bloco;
- custo: usa o `stress_net_bps` já materializado nos trades congelados.

## Safety boundary

O Fast64 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast64-session-side-filter-walkforward/report.json`;
- holdouts: `backend/data/microstructure_v1/reports/fast64-session-side-filter-walkforward/holdouts.json`;
- baseline por bloco: `backend/data/microstructure_v1/reports/fast64-session-side-filter-walkforward/baseline_by_block.json`.

Hashes:

- report: `058c2b38ecff87bdc65d2739a6c42861ec97abe48d11a075e5da2bc6ffea029e`;
- holdouts: `b0ac1bfc2fe954a0a20c19c44a5ab76213ecb3a3dcc794f60648e38c2fe95ed3`;
- baseline: `684feab12769a6c47004b3626f5cfa4559ee365c2a60b586440d899049697558`.

## Resultado

Status:

`FAST64_SESSION_SIDE_FILTER_FAILED_WALKFORWARD`

Resumo:

- blocos: 11;
- trades originais avaliados: 2058;
- trades após filtros selecionados por walk-forward: 202;
- holdouts positivos em stress: `3/11`;
- stress total: `-555.9178770498207` bps;
- expected total: `-353.9178770498208` bps;
- pior holdout: `-500.1104485672937` bps.

Filtros escolhidos:

- Fast43: `side=SELL&utc_18_23`;
- Fast44–Fast53: `utc_00_05`.

Por holdout:

| Holdout | Filtro congelado | Trades | Stress total | Passou |
| --- | --- | ---: | ---: | --- |
| Fast43 | `side=SELL&utc_18_23` | 2 | `-7.12883364617784` | não |
| Fast44 | `utc_00_05` | 4 | `35.599677815967134` | sim |
| Fast45 | `utc_00_05` | 1 | `9.620339786035483` | não, amostra |
| Fast46 | `utc_00_05` | 12 | `-92.08171616982939` | não |
| Fast47 | `utc_00_05` | 3 | `21.48031580460063` | sim |
| Fast48 | `utc_00_05` | 1 | `3.642703241589925` | não, amostra |
| Fast49 | `utc_00_05` | 140 | `-500.1104485672937` | não |
| Fast50 | `utc_00_05` | 31 | `-63.56243535188421` | não |
| Fast51 | `utc_00_05` | 1 | `7.995471442005238` | não, amostra |
| Fast52 | `utc_00_05` | 2 | `21.51775320208043` | não, amostra |
| Fast53 | `utc_00_05` | 5 | `7.109295393085544` | sim |

## Interpretação

O treino encontrou quase sempre `utc_00_05`, mas isso não generalizou. O maior
alerta é Fast49: o mesmo filtro escolhido por treino virou `-500.11` bps em uma
semana com volume de trades suficiente. Isso caracteriza filtro de sessão
instável, não estratégia.

## Conclusão

Não há estratégia aprovada. O caminho “talvez só horário/lado/probabilidade
resolva” foi testado de forma causal e falhou. A próxima busca precisa de outra
família de sinal ou de features que expliquem regime antes da entrada com mais
força do que sessão/lado simples.
