# Fast66 — auditoria do gate liquidez + volatilidade

Fast65 indicou que a pista mais forte não era horário/lado, mas regime de
microestrutura ativa. Fast66 restringe a busca para uma família menor e mais
interpretável:

> operar o top-p congelado apenas quando `quote_volume_30s` está alto E
> `realized_vol_30s_bps` está alto.

Resultado: é a evidência mais promissora até agora, mas ainda não é estratégia.

## Escopo

- fonte: dataset mergeado do Fast65;
- mercado: BTCUSDT perpetual;
- blocos: Fast43–Fast53;
- gate fixo principal: `quote_volume_30s >= q0.80` e
  `realized_vol_30s_bps >= q0.75`, sempre calculado só nos blocos de treino;
- alternativa avaliada: seleção nested restrita a AND de liquidez+volatilidade;
- execução: one-position;
- custo: `stress_net_bps` materializado nos trades congelados.

## Safety boundary

O Fast66 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast66-liquidity-volatility-gate-audit/report.json`;
- holdouts gate fixo: `backend/data/microstructure_v1/reports/fast66-liquidity-volatility-gate-audit/fixed_q80_q75_holdouts.json`;
- holdouts seleção nested: `backend/data/microstructure_v1/reports/fast66-liquidity-volatility-gate-audit/selected_strict_and_holdouts.json`.

Hashes:

- report: `36bd8951ec87ad10523706e2212238292b8fd1241787eaeb48e4250a0252606d`;
- fixed q80/q75: `ca9d9e4e3236c0e46fa675681f6f670b1f0886925443dd87de58a253411c5c68`;
- selected strict AND: `cd0e9b68f257b3e7576e808a222e1b9bbc2bcf10fecdbb18323d83678a35b92a`.

## Resultado principal

Status:

`FAST66_LIQUIDITY_VOLATILITY_GATE_PROMISING_BUT_FAILED_ROBUSTNESS`

Gate fixo `qv q0.80 + vol q0.75`:

- trades: 304;
- holdouts positivos em stress: `8/11`;
- stress total: `430.82895662120455` bps;
- expected total: `734.8289566212046` bps;
- pior holdout: `-9.511969539767648` bps.

Seleção nested strict AND:

- trades: 298;
- holdouts positivos em stress: `7/11`;
- stress total: `415.88319853069856` bps;
- pior holdout: `-9.511969539767648` bps.

O gate fixo foi melhor que a seleção nested.

## Por holdout — gate fixo

| Holdout | Trades | Stress total | Stress mean | Passou |
| --- | ---: | ---: | ---: | --- |
| Fast43 | 16 | `36.80019220145827` | `2.300012012591142` | sim |
| Fast44 | 17 | `39.91916204736644` | `2.348186002786261` | sim |
| Fast45 | 7 | `36.196131269095204` | `5.170875895585029` | sim |
| Fast46 | 11 | `47.332596597180256` | `4.3029633270163865` | sim |
| Fast47 | 2 | `31.70727239686306` | `15.85363619843153` | não, amostra |
| Fast48 | 6 | `12.652793401086841` | `2.10879890018114` | sim |
| Fast49 | 137 | `75.29531378768176` | `0.5496008305670201` | sim |
| Fast50 | 25 | `140.8820834699975` | `5.6352833387998995` | sim |
| Fast51 | 30 | `23.531026683001656` | `0.7843675561000552` | sim |
| Fast52 | 33 | `-9.511969539767648` | `-0.28824150120508024` | não |
| Fast53 | 20 | `-3.97564569275878` | `-0.198782284637939` | não |

## Interpretação

Essa é a primeira família que:

- fica fortemente positiva no agregado;
- transforma o desastre do Fast49 em resultado positivo;
- usa um gate causal simples e operacionalmente interpretável;
- reduz a tese para "só operar quando há liquidez e volatilidade recentes".

Mas ainda falha a barra de estratégia:

- Fast52 e Fast53 continuam negativos;
- Fast47 tem só 2 trades, abaixo do mínimo;
- a evidência é histórica, não prospectiva;
- ainda não há shadow live-like desse gate.

## Decisão

Não aprovar estratégia. Não Testnet. Não LIVE.

Próximo passo correto: pré-registrar Fast67 como candidato de pesquisa:

- gate fixo `quote_volume_30s >= q0.80` AND
  `realized_vol_30s_bps >= q0.75`;
- congelar a regra antes de olhar novos WALs;
- rodar em prospective/shadow, com mínimo de dias/trades;
- reprovar automaticamente se qualquer bloco novo ficar negativo em stress com
  amostra suficiente.
