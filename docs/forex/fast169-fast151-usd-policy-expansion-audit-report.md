# Fast169 — Fast151 USD policy-expansion audit

Fast169 expandiu a pergunta do Fast168. Em vez de testar só `hold5/10/20` no
mesmo basket Fast151, ele auditou a família USD-quoted inteira já materializada:

- subsets de `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- canais Donchian `20`, `55`, `100`;
- holds `5`, `10`, `20`;
- `SHORT` only;
- `stop2_fixed_1r`;
- máximo 1 posição aberta por par;
- circuit breaker `dd800_cooldown180d`.

Status:

`FAST169_FAST151_UNIQUE_STRICT_SURVIVOR_IN_USD_POLICY_EXPANSION`

## Resposta curta

Dentro dessa família, só uma combinação passa pré-2019 e 2019-2026:

`AUDUSD+EURUSD+GBPUSD+NZDUSD:d1_donchian_fade_ch100_hold20`

Ou seja: a Fast151 não é só “uma candidata entre várias parecidas”. Nessa
expansão de basket/canal/hold, ela foi a única sobrevivente estrita.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| combinações avaliadas | `135` |
| passaram pré-2019 | `2` |
| passaram current 2019-2026 | `26` |
| passaram os dois blocos | `1` |

## Sobrevivente estrita

| Basket | Policy | Pré LCB | Current LCB | Current swap BE | Decisão |
|---|---|---:|---:|---:|---|
| `AUDUSD+EURUSD+GBPUSD+NZDUSD` | `d1_donchian_fade_ch100_hold20` | `+32.2878` | `+9.6335` | `0.9589` | manter candidata |

## Por que isso importa

Isso fecha uma dúvida grande: talvez existisse outro basket menor, canal menor
ou hold menor que mantivesse a edge e melhorasse o problema de swap.

O audit encontrou vários setups bonitos no current, principalmente `hold5`, mas
eles não passam o bloco pré-2019. Exemplo:

| Basket | Policy | Current LCB | Current swap BE | Falha no pré |
|---|---|---:|---:|---|
| `AUDUSD+EURUSD+GBPUSD+NZDUSD` | `ch100_hold5` | `+19.1220` | `5.6610` | LCB negativo, anos ruins, positive-year share baixo |
| `AUDUSD+EURUSD+GBPUSD` | `ch100_hold5` | `+18.0368` | `5.5593` | trade count baixo, mean negativo, LCB negativo, anos ruins |
| `AUDUSD+EURUSD+NZDUSD` | `ch100_hold5` | `+18.8608` | `5.4318` | trade count baixo, LCB negativo |
| `AUDUSD+GBPUSD+NZDUSD` | `ch100_hold5` | `+15.7377` | `4.6207` | trade count baixo, anos ruins, positive-year share baixo |

Leitura: o mercado até oferece “atalhos bonitos”, mas eles aparecem quando a
gente olha o período atual. O gate cronológico impede transformar isso em regra
operável.

## O que isso muda no caminho

Fast169 muda o diagnóstico:

- não falta testar basket/canal/hold simples nessa família USD-quoted;
- a candidata Fast151 ficou mais especial, porque é única sobrevivente estrita;
- mas ela continua não aprovada, porque ainda depende de:
  - M1 pós-pré-registro;
  - observação prospectiva real;
  - tabela de swap da corretora;
  - autorização explícita para qualquer estágio de execução.

## Segurança

Fast169 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast169_fast151_usd_policy_expansion_audit.py`;
- teste:
  `backend/tests/unit/test_fast169_fast151_usd_policy_expansion_audit.py`;
- report JSON:
  `backend/data/lab_fast169/fast151-usd-policy-expansion-audit/report.json`;
- métricas:
  `backend/data/lab_fast169/fast151-usd-policy-expansion-audit/usd-policy-expansion-metrics.csv`;
- resumo:
  `backend/data/lab_fast169/fast151-usd-policy-expansion-audit/usd-policy-expansion-summary.json`.

Hashes:

- report:
  `2a6651fca71108f4e69002665b5588e0a4ea55fbab1081f9468f49461f92851d`;
- runner:
  `ce23f03155f8b6c8e68f09be45a828ce43a830f0154c47f04a1993e74d219957`;
- teste:
  `be970d2d161721ac423bc2295b895dea1e904ebfb8430e84c1634312ba5ddac7`;
- métricas:
  `72662c468f5560f7b8c1b1011f46c1aea1cc3728f22be28722940a873552b63e`;
- resumo:
  `ce417b1b71bd8238d806c7d4d41cff6aa6215d5032f693dbb07a0e9b660b98bd`.
