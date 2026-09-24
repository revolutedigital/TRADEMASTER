# Fast106 — scout curto de horizonte maior e alvo maior

Resposta objetiva: não temos uma estratégia. O Fast106 testou a nova hipótese
de operar menos, mirar movimentos maiores e dar mais tempo para o trade andar.
Na amostra curta, apareceram sobreviventes no screen, mas eles falharam quando
revalidados por walk-forward nos blocos históricos.

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

## Hipótese testada

Fast105 matou a linha de veto fixo de micro-retorno. Fast106 muda a família:

- horizonte: `900s`;
- alvos testados: `8`, `12`, `16` e `24` bps;
- custo explícito: `2` bps;
- stop inicial: `20` bps;
- stride de decisão: `300s`;
- top fractions: `0.5%`, `1%`, `2%`, `5%`;
- dados: `BTCUSDT` normalized `aggTrades`, sem book/fill model.

Este primeiro run foi deliberadamente pequeno para evitar gastar tempo demais
num full scout antes de ver se a família sobrevivia ao walk-forward:

- históricos recentes: `research-lowcost-fast51-holdout`,
  `research-lowcost-fast52-holdout`, `research-lowcost-fast53-holdout`;
- extensão: `research-lowcost-fast56-holdout-new`;
- `1` dia mais recente por bloco.

## Resultado

Status do runner:

`FAST106_LONG_HORIZON_FAILED_WALKFORWARD_OR_EXTENSION`

Resumo:

| Métrica | Valor |
|---|---:|
| Combinações no screen | `16` |
| Sobreviventes no screen | `3` |
| Sobreviventes aprovados no walk-forward + extensão | `0` |
| Estratégia aprovada | `false` |

O ponto importante: o screen encontrou três configurações aparentemente boas,
mas todas quebraram quando o modelo foi re-treinado deixando blocos históricos
fora.

## Sobreviventes que falharam

| Horizonte | Alvo | Top fraction | Screen histórico | Screen extensão | Walk-forward histórico |
|---|---:|---:|---:|---:|---:|
| `900s` | `8` bps | `0.5%` | `+31.628458613314912` bps / `6` trades | `+36.0` bps / `6` trades | `-12.0` bps / `12` trades |
| `900s` | `8` bps | `1%` | `+35.76292833892716` bps / `15` trades | `+22.0` bps / `13` trades | `-37.23145150538526` bps / `22` trades |
| `900s` | `24` bps | `5%` | `+95.87403751476755` bps / `70` trades | `+5.9878291641857855` bps / `27` trades | `-373.7990706726452` bps / `78` trades |

## Decisão

Fast106 smoke não é estratégia e não deve ir para Testnet.

O teste mostrou a mesma doença central dos runs anteriores: uma seleção parece
boa quando treinada/avaliada de forma agregada, mas a estabilidade por bloco não
aparece. O trailing/tempo maior não salva se o modelo está pegando um padrão
frágil ou localizado demais.

## O que isso ensina

- A rota “top-p + alvo maior + horizonte maior” ainda pode ser investigada, mas
  o primeiro filtro não demonstrou robustez.
- A métrica que mais importa agora é estabilidade por bloco, não lucro agregado
  no screen.
- Se insistirmos nessa família, o próximo passo precisa ser otimizar o runner
  para não recarregar `aggTrades` repetidamente e só então rodar um full scout.
- Não há base matemática suficiente aqui para automatizar entradas reais.

## Próximo caminho recomendado

O melhor próximo passo não é operar. É melhorar a infraestrutura do scout:

1. cachear carregamento de `aggTrades` por data/horizonte;
2. rodar Fast106 completo com `900s` e `1800s`, alvos `8/12/16/24`;
3. exigir que qualquer candidato vença walk-forward por bloco antes de qualquer
   pré-registro prospectivo;
4. se nenhum sobreviver, abandonar esta família e migrar para outro mercado,
   outro par ou dados com book/fill mais rico.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python backend/scripts/research/run_fast106_long_horizon_lowcost_scout.py \
  --output-root backend/data/microstructure_v1/reports/fast106-long-horizon-lowcost-scout-smoke \
  --horizons-seconds 900 \
  --target-bps-values 8,12,16,24 \
  --historical-root-limit 3 \
  --max-days-per-root 1 \
  --progress
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast106-long-horizon-lowcost-scout-smoke/report.json`;
- report sha256:
  `2738d1ebb824ce6756d0fc368eb8e833a3e0588367e5ec168f89bd29fd41972f`;
- screen results sha256:
  `8d65cdb45b4c0d20d990e94b79292a4c1f1f554f3110b52b24365e95f0e5fceb`;
- validations sha256:
  `9b746177a151a4b7c2a3947662ff80d38a4c12c9d83e3015e8e58d4a222201e5`;
- row manifest sha256:
  `96e182fa2cbf5cccd62dd92cefbaaff2dda5a7ad1e06bb324406c582be70ce5d`.
