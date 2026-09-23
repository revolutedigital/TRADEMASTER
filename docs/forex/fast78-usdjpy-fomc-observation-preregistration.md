# Fast78 — pré-registro de observação prospectiva USDJPY/FOMC

Registrado em 2026-09-23, depois do Fast77 e antes de qualquer evento futuro
ser usado para avaliar esta hipótese. Este documento congela a hipótese fraca
que sobrou. Ele não autoriza Demo, Testnet, LIVE, envio de ordem, commit de
ledger ou operação manual.

## Decisão de partida

Fast77 encontrou um único weak survivor com estabilidade temporal na auditoria
2024-2026, mas sem significância:

- par: `USDJPY`;
- evento: `FOMC rate decision`;
- política: `drift_thr2_d5_h120_trail_1r_0_75r`;
- seleção pré-2024: `31` trades, stress médio `+0.031920612903021145` pips;
- seleção por ano positiva em stress: `2/5`;
- auditoria 2024-2026: `20` trades, stress médio `+4.4120315000005235` pips;
- lower stress sem correção: `-1.9468284046199846` pips;
- lower stress Bonferroni: `-7.488511433014544` pips.

Conclusão de partida: hipótese observável, não estratégia.

## Safety boundary

Durante todo o Fast78:

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
- `event_name=FOMC rate decision`;
- `policy_id=drift_thr2_d5_h120_trail_1r_0_75r`;
- custos: avaliar sempre em `net_base_pips` e `net_stress_pips`, com decisão
  primária pelo stress;
- implementação da política: a mesma semântica usada no artefato fechado do
  Fast13/Fast76/Fast77. O `policy_id` não pode ser reinterpretado manualmente
  para produzir entrada diferente;
- dados: M1 e calendário oficial no mesmo padrão Fast13.

Nenhum outro par, evento, threshold, horizonte ou trailing entra no Fast78.

## Janela prospectiva

São elegíveis apenas eventos com `event_timestamp` posterior ao commit deste
documento. Qualquer evento entre `2026-09-01` e este pré-registro fica fora,
mesmo que os dados sejam baixados depois.

A avaliação prospectiva não pode usar eventos futuros para alterar:

- política;
- custos;
- par;
- nome do evento;
- horizonte;
- stop/trailing;
- filtros mínimos;
- regra de aprovação.

## Regras de observação

Para cada FOMC elegível:

1. Gerar exatamente uma decisão observacional para `USDJPY` usando a política
   congelada.
2. Salvar entrada, saída, `net_base_pips`, `net_stress_pips`,
   `exit_base_reason`, `exit_stress_reason` e timestamps.
3. Marcar explicitamente `research_only=true` e `orders_sent=false`.
4. Não enviar ordem, não abrir Testnet e não simular preenchimento como se fosse
   broker real.

## Gates

### Gate de falha precoce

A hipótese é encerrada como falha se qualquer condição ocorrer:

- após os primeiros `8` eventos elegíveis, `total_stress_pips <= -20`;
- dois anos-calendário prospectivos consecutivos com stress médio negativo;
- qualquer alteração necessária na política congelada para conseguir calcular
  o resultado.

### Gate mínimo para virar candidato de shadow offline

Mesmo que o resultado pareça bom antes, não há promoção antes de:

- pelo menos `20` eventos prospectivos elegíveis;
- stress médio > `+2.0` pips/trade;
- base médio > `0`;
- lower confidence bound unilateral de stress, sem correção, > `0`;
- pelo menos `60%` dos anos prospectivos com stress médio > `0`;
- nenhuma regra alterada desde este documento.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast78 não tem gate de Testnet. Testnet exigiria um novo pré-registro, um novo
relatório, revisão dos riscos de execução real em calendário macro e autorização
explícita do Igor.

## O que NÃO será feito

- Trocar CPI por FOMC ou FOMC por CPI depois de ver eventos novos.
- Adicionar outro par porque funcionou melhor.
- Relaxar o limite inferior se o total ficar bonito.
- Usar evento entre `2026-09-01` e o commit deste documento como se fosse
  prospectivo.
- Chamar o resultado de estratégia antes de um relatório futuro passar os gates.

## Próxima implementação

O próximo passo técnico, quando houver dados prospectivos suficientes ou quando
quisermos deixar a coleta pronta, é criar um runner Fast78 que:

- lê o calendário prospectivo;
- aplica somente a hipótese congelada;
- grava JSONL de observação local;
- recusa qualquer ordem;
- calcula os gates acima sem recalibrar nada.
