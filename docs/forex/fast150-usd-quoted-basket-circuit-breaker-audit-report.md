# Fast150 — USD-quoted basket circuit-breaker audit

Fast150 testou a próxima peça operacional em cima da candidata Fast149:

> Dá para pausar entradas em regime ruim sem destruir o edge?

Status:

`FAST150_CIRCUIT_BREAKER_CANDIDATE_REQUIRES_PREREGISTRATION`

## Resposta curta

Sim. A melhor regra selecionada no pré-2019 foi:

`dd800_cooldown180d`

Regra do breaker:

- manter a entrada congelada;
- manter `SHORT`;
- manter `stop2_fixed_1r`;
- manter a cesta `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- manter o overlay Fast149 `max_per_pair_1`;
- calcular drawdown por equity fechada, usando apenas trades já encerrados;
- se o drawdown fechado bater `-800 pips`, pausar novas entradas por `180 dias`;
- trades já abertos continuam até o gerenciamento normal.

## Comparação

| Amostra | Versão | Trades | Média stress | LCB stress | Total stress | Max drawdown | Max concorrência | Anos negativos seguidos |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Pré-2019 | Fast149 sem breaker | `58` | `+52.0126` | `+18.1781` | `+3016.7295` | `-1193.7464` | `4` | sim |
| Pré-2019 | Fast150 `dd800_cooldown180d` | `51` | `+67.4106` | `+32.2878` | `+3437.9421` | `-900.2085` | `4` | não |
| 2019–2026 | Fast149 sem breaker | `86` | `+33.6173` | `+9.6335` | `+2891.0872` | `-616.6221` | `4` | não |
| 2019–2026 | Fast150 `dd800_cooldown180d` | `86` | `+33.6173` | `+9.6335` | `+2891.0872` | `-616.6221` | `4` | não |

No pré-2019, o breaker pulou `7` trades e teve `1` gatilho. No período
2019–2026, ele não foi acionado: `0` trades pulados e `0` gatilhos.

## Interpretação

Esse é o primeiro pacote que parece operacionalmente coerente:

- o edge continua positivo no pré e no atual;
- o LCB fica positivo nos dois períodos;
- o drawdown histórico cai para dentro do limite testado de `1000 pips`;
- a simultaneidade fica travada em `4`;
- o problema dos anos negativos consecutivos no pré-2019 desaparece;
- o período atual não foi “salvo” pelo breaker — ele já passava com o overlay
  Fast149 e permaneceu intacto.

Leitura honesta:

`max_per_pair_1 + dd800_cooldown180d` deve virar a candidata operacional
pré-registrada. Ainda não é uma estratégia aprovada, porque o breaker foi
escolhido olhando histórico. O próximo passo correto é congelar essa política
em novo pré-registro e observar somente dados posteriores ao pré-registro.

## Segurança

Fast150 é pesquisa offline:

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
  `backend/scripts/research/run_fast150_usd_quoted_basket_circuit_breaker_audit.py`;
- teste:
  `backend/tests/unit/test_fast150_usd_quoted_basket_circuit_breaker_audit.py`;
- report JSON:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/report.json`;
- métricas:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/circuit-breaker-metrics.csv`;
- trades pré selecionados:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-pre-trades.csv`;
- trades current selecionados:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-current-trades.csv`;
- skips pré:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-pre-skipped.csv`;
- skips current:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-current-skipped.csv`;
- triggers pré:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-pre-triggers.csv`;
- triggers current:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-current-triggers.csv`;
- equity pré:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-pre-equity-curve.csv`;
- equity current:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/selected-breaker-current-equity-curve.csv`;
- resumo:
  `backend/data/lab_fast150/usd-quoted-basket-circuit-breaker-audit/circuit-breaker-summary.json`.

Hashes:

- report:
  `d3d3f98c60ffc74335ecaa8db8eeb79603dc9dc68574e14a998d362755f0019e`;
- métricas:
  `77bb1d195e7dd27424d5b4bef9ea9a4922fbc4f760f93752aa2c56e55bd4aaab`;
- trades pré:
  `e619dc3e28f50523a3b8440c15078ab4b97c0ffc4aaebc7eafef0e44c54598e5`;
- trades current:
  `437af9f105fa677ae8caea6ed9088666f9fbc64a14d15459f61716659efd85a0`;
- skips pré:
  `dbaf6085267bcf176acdc697fddb7c69e561107ee014d190349950e6a74546f4`;
- skips current:
  `65441c313def440e1ae71024bca1de08769e1cf852c4a3214f4a966488ad7381`;
- triggers pré:
  `40163ffead15c81736dc63ed0cb2c531206f79d8e7c3ea03991935b7450d6b7d`;
- triggers current:
  `81fd67ec37638a709fbe3e4188d9b538d5cb73cc8c437ca691c77f006c030551`;
- equity pré:
  `d54b99e05e39bf1659661795ec0f49d1996e57ab308104b396168ffcaa844117`;
- equity current:
  `33e87507767c6ad322aa5a618cc8c27d8de07fd78fbe91deb5d7ec10190a3387`;
- resumo:
  `94a08c6f73ce7360f13b402eb71408a922b6d682666eb9a4aa40f6de4d2880c9`.
