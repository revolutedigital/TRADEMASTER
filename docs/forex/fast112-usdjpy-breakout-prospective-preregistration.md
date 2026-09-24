# Fast112 — pré-registro prospectivo USDJPY London breakout

Registrado depois do Fast111 e antes de qualquer dado futuro ser usado para
avaliar esta hipótese. Este documento congela uma única regra observacional. Ele
não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou operação
manual.

## Decisão de partida

Fast111 encontrou um filtro weak para a linha que sobrou do Fast110:

- par: `USDJPY`;
- família: `F1a / London breakout`;
- configuração: `F1a.rr2.5.w0.6-2.0.se11:00.x16:30`;
- filtro: `london_entry_hour=8 && stop_bucket=wide`;
- bucket `wide` congelado pelo S0 em `stop_pips > 94.88491666666664`;
- S0: `26` trades stress, média `+0.093614953450405 R`, anos positivos `4/5`;
- S1: `26` trades stress, média `+0.02441183149637864 R`, anos positivos `3/5`;
- S2: `36` trades stress, média `+0.10626468772283898 R`, anos positivos `2/3`;
- strict survivors no Fast111: `0`.

Conclusão de partida: hipótese observável, não estratégia.

## Safety boundary

Durante todo o Fast112:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

Se qualquer implementação futura tentar enviar ordem ou gravar ledger real com
base neste pré-registro, isso é defeito de segurança.

## Hipótese congelada

A única hipótese observada será:

- `pair=USDJPY`;
- `config=F1a.rr2.5.w0.6-2.0.se11:00.x16:30`;
- semântica da estratégia: a mesma do Fast2/Fast111 para `session_breakout`;
- janela de range: `00:00-08:00` Londres;
- janela de sinal: `08:00-11:00` Londres;
- saída: `16:30` Londres;
- filtro de range original: `0.6-2.0 ATR`;
- reward/risk: `2.5`;
- filtro adicional congelado:
  - `london_entry_hour == 8`;
  - `stop_pips > 94.88491666666664`;
- decisão primária: `stress_r`;
- métrica secundária: `base_r`.

Nenhum outro par, reward/risk, hora, bucket, threshold, dia da semana, lado,
trailing ou filtro de volatilidade entra no Fast112.

## Janela prospectiva

São elegíveis apenas entradas com `entry_time` posterior ao timestamp UTC do
commit que registrou este documento.

Qualquer trade histórico usado em Fast111 fica fora, mesmo que seja
reprocessado depois. A avaliação prospectiva não pode usar entradas futuras para
alterar:

- configuração;
- threshold de stop;
- custos;
- filtro;
- gate;
- critério de aprovação.

## Regras de observação

Para cada entrada elegível:

1. Gerar exatamente uma decisão observacional para `USDJPY` usando a regra
   congelada.
2. Salvar entrada, saída, lado, `base_r`, `stress_r`, `stop_pips`, `entry_time`,
   `exit_time` e motivo de saída.
3. Marcar explicitamente `research_only=true` e `orders_sent=false`.
4. Não enviar ordem, não abrir Testnet e não simular preenchimento como se fosse
   broker real.

## Gates

### Gate de falha precoce

A hipótese é encerrada como falha se qualquer condição ocorrer:

- após os primeiros `12` trades elegíveis, `total_stress_r <= -3.0`;
- dois anos-calendário prospectivos consecutivos com stress médio negativo;
- qualquer alteração necessária na regra congelada para conseguir calcular o
  resultado.

### Gate mínimo para virar candidato de shadow offline

Mesmo que o resultado pareça bom antes, não há promoção antes de:

- pelo menos `50` trades prospectivos elegíveis;
- stress médio `> +0.05 R/trade`;
- base médio `> 0`;
- lower confidence bound unilateral de stress, sem correção, `> 0`;
- pelo menos `60%` dos anos prospectivos com stress médio `> 0`;
- nenhuma regra alterada desde este documento.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast112 não tem gate de Testnet. Testnet exigiria novo relatório, revisão de
risco de execução real, release research-only válida e autorização explícita do
Igor.

## O que NÃO será feito

- Trocar `USDJPY` por outro par.
- Trocar reward/risk porque outro pareceu melhor.
- Relaxar `stop_pips > 94.88491666666664`.
- Incluir entrada às 09h/10h Londres.
- Adicionar filtro de dia da semana depois de ver os próximos trades.
- Chamar o resultado de estratégia antes de relatório futuro passar os gates.

## Próxima implementação

O próximo passo técnico é criar um runner Fast112 que:

- valida o hash deste pré-registro;
- filtra somente a hipótese congelada;
- grava JSONL de observação local;
- recusa qualquer ordem;
- calcula os gates acima sem recalibrar nada.
