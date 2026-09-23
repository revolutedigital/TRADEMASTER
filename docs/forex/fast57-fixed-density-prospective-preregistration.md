# Fast57 — pré-registro do gate fixo em dados prospectivos

Data: 2026-09-23  
Status: pré-registrado antes de abrir qualquer dado posterior a 2026-09-22  
Escopo: pesquisa offline, sem Testnet/LIVE.

## Candidato congelado

O candidato a partir daqui fica fixo:

- modelo: artefato Fast43;
- model hash: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- target original: `latency_paid_stress_before_stop`;
- horizonte: 300 s;
- features: `flow_price_session`;
- top-p: 1%;
- threshold: `0,90195311025804`;
- trailing: `micro_8_3`;
- gate: `recent_300s_lte_0`;
- custo esperado: 1 bps roundtrip;
- custo stress: 2 bps roundtrip;
- latência: 100 ms;
- uma posição por vez.

Não é permitido:

- mexer no threshold;
- mexer no gate;
- trocar trailing;
- trocar horizonte/features;
- escolher subperíodo depois de ver resultado;
- promover para Testnet/LIVE com esse pré-registro.

## Fonte

Usar apenas Binance public archive `BTCUSDT` futures `aggTrades` diário, com
manifest `VALID`, `sequence_gap_count=0`, e normalização registrada.

Primeiro dia elegível: 2026-09-23 UTC, quando o arquivo diário completo estiver
disponível.

## Janela mínima

Fast57 só pode ser avaliado quando tiver, no mínimo:

- 10 dias UTC completos novos; e
- 100 trades executados pelo gate.

Se 10 dias não gerarem 100 trades, continuar acumulando dias até atingir 100
trades ou 30 dias, o que vier primeiro.

## Critérios de leitura

Para considerar que o candidato merece o próximo gate de book/shadow:

1. stress total > 0 bps;
2. stress médio > 0 bps/trade;
3. pelo menos 60% dos dias com trades positivos em stress;
4. máximo drawdown em stress menor que o lucro total em stress;
5. resultado com gate melhor que o mesmo candidato sem gate;
6. nenhum bloco semanal com perda maior que 2 vezes o ganho médio semanal.

Falhar em qualquer item rejeita Fast57 como confirmação.

Passar todos os itens não aprova estratégia. Passar só autoriza:

- replay com book/quote quando disponível; ou
- shadow prospectivo research-only sem ordem real.

## Segurança

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

Sem Testnet/LIVE.

