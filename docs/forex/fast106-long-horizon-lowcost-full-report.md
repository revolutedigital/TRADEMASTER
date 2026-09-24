# Fast106 full — horizonte maior e alvo maior reprovados no screen

Resposta objetiva: esta família não virou estratégia.

O smoke do Fast106 já tinha falhado no walk-forward. O run full, com todos os
blocos históricos disponíveis, foi mais conclusivo: nenhuma combinação passou
sequer pelo screen inicial.

## Safety boundary

Mantido em todos os passos:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## O que mudou antes do full

O primeiro runner era correto, mas ineficiente: ele relia os mesmos
`aggTrades` para cada alvo. O runner foi ajustado para carregar cada
dia/horizonte uma vez e reaproveitar os trades para todos os alvos.

Validação da otimização:

- teste unitário novo confirmou que múltiplos alvos no mesmo dia/horizonte
  fazem apenas um carregamento;
- o smoke otimizado reproduziu o mesmo resultado estatístico do smoke anterior;
- tempo observado do smoke caiu de aproximadamente `24.6s` para `9.3s`.

## Hipótese full testada

- horizontes: `900s` e `1800s`;
- alvos: `8`, `12`, `16` e `24` bps;
- top fractions: `0.5%`, `1%`, `2%`, `5%`;
- custo explícito: `2` bps;
- stop inicial: `20` bps;
- stride de decisão: `300s`;
- dados: `BTCUSDT` normalized `aggTrades`, sem book/fill model;
- históricos: todos os `research-lowcost-fast*` elegíveis, de Fast42 a Fast53;
- extensão: `research-lowcost-fast56-holdout-new`.

## Resultado

Status do runner:

`FAST106_LONG_HORIZON_FAILED_SCREEN`

Resumo:

| Métrica | Valor |
|---|---:|
| Combinações testadas | `32` |
| Sobreviventes no screen | `0` |
| Validações walk-forward | `0` |
| Estratégia aprovada | `false` |

Critério de sobrevivência do screen:

- extensão com pelo menos `3` trades;
- extensão positiva;
- histórico positivo.

Nenhuma combinação satisfez isso.

## Melhores por histórico, ainda negativas

Mesmo as combinações menos ruins no histórico foram negativas:

| Horizonte | Alvo | Top fraction | Histórico | Extensão |
|---|---:|---:|---:|---:|
| `1800s` | `8` bps | `0.5%` | `-47.73782942586462` bps / `153` trades | `-30.0` bps / `9` trades |
| `900s` | `12` bps | `1%` | `-68.56763028047938` bps / `317` trades | `-30.023639639397707` bps / `23` trades |
| `1800s` | `24` bps | `0.5%` | `-116.19885804193412` bps / `129` trades | `-22.0` bps / `3` trades |
| `900s` | `24` bps | `0.5%` | `-117.58810777019389` bps / `154` trades | `-32.75828975856534` bps / `6` trades |
| `900s` | `8` bps | `0.5%` | `-143.47105656283358` bps / `139` trades | `-2.0` bps / `9` trades |

## Melhores por extensão também não servem

Algumas combinações foram positivas na extensão, mas todas tinham histórico
negativo forte. Isso não é edge; é o mesmo padrão de instabilidade:

| Horizonte | Alvo | Top fraction | Histórico | Extensão |
|---|---:|---:|---:|---:|
| `900s` | `24` bps | `2%` | `-435.1986377426227` bps / `644` trades | `+35.70064662565263` bps / `30` trades |
| `1800s` | `12` bps | `1%` | `-290.96717893052016` bps / `317` trades | `+10.724756515911112` bps / `15` trades |
| `900s` | `24` bps | `5%` | `-1917.930570227496` bps / `1505` trades | `+8.080044079754238` bps / `54` trades |
| `1800s` | `12` bps | `2%` | `-233.43898509912674` bps / `588` trades | `+6.0` bps / `23` trades |

## Decisão

Encerrar a família Fast106 como não estratégia.

O resultado mata a hipótese de que bastaria aumentar horizonte/alvo e reduzir
frequência mantendo a mesma família de features/top-p. Quando a amostra aumenta,
o sinal não fica mais sólido; ele desaparece antes mesmo do walk-forward.

## O que isso ensina

- Trailing/stop não cria edge se a entrada não tem expectativa positiva.
- “Andar algum tanto e proteger” só funciona se a entrada tiver assimetria real
  antes do custo.
- Horizonte maior reduziu ruído em tese, mas não criou estabilidade matemática.
- Top-p continua insuficiente quando o modelo prevê payoff frágil em vez de um
  regime estrutural.

## Próximo caminho recomendado

Não continuar tunando Fast106.

Se a busca continuar, a próxima família precisa mudar alguma coisa estrutural:

1. outro mercado/par com microestrutura melhor;
2. dados com book/fill/slippage mais ricos, não apenas `aggTrades`;
3. modelo de regime antes do modelo de entrada;
4. alvo explícito de “grandes movimentos raros” com validação por bloco desde o
   primeiro screen;
5. abandonar BTCUSDT low-cost microstructure se a próxima família também falhar.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python backend/scripts/research/run_fast106_long_horizon_lowcost_scout.py \
  --output-root backend/data/microstructure_v1/reports/fast106-long-horizon-lowcost-scout-full \
  --progress
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast106-long-horizon-lowcost-scout-full/report.json`;
- report sha256:
  `4bb1330e8002340d1012b17f999af25b6f626809a91cae0471d52d77f3585cbd`;
- screen results sha256:
  `3b2f7153f110a4417fef752343838d3d528bc6f4f66fdef96ae8ae47e80aad45`;
- validations sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- row manifest sha256:
  `ea5a56fd75b7f09792a42dbf27cb888bcae0d9835744c22656b89de46dfd5c0b`.
