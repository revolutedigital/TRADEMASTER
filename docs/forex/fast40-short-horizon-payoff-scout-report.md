# Fast40 — scout de payoff em horizonte curto

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32, Fast35, Fast36  
Tese: buscar trades mais curtos, com target/trailing pequenos e seleção top-p  
Resultado: falhou replicação.

## Por que esta rodada existe

Depois de Fast37/Fast39, a pergunta ficou mais específica:

> "E se o edge não estiver em carregar mais tempo, mas em capturar um deslocamento
> pequeno e proteger rápido com trailing?"

Fast40 testa exatamente isso: horizonte de 30 segundos, decisões a cada 5
segundos, latência simulada de 100 ms, targets entre 1 e 5 bps, trailing entre
1 e 5 bps, e seleção por top-p do payoff executável.

## Método

Para cada janela de holdout:

1. remove a janela de teste;
2. treina o modelo nas demais janelas;
3. estima o payoff executável do trailing/stop;
4. seleciona apenas o topo da distribuição prevista;
5. aplica o modelo congelado na janela segurada fora;
6. mede o resultado independente e one-position.

Configuração:

- horizonte: 30 segundos;
- stride de decisão: 5 segundos;
- latência: 100 ms;
- stop inicial: 20 bps;
- targets testados: 1, 2, 3, 5 bps;
- trailing testados: 1, 2, 3, 5 bps;
- top-p testados: 2%, 5%, 10%, 20%;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- mínimo por janela: 5 execuções.

## Fonte

Report:

`backend/data/microstructure_v1/reports/book-payoff-model-scout-fast40-30s-short-horizon-100ms/report.json`

Hash:

`170f018650d923528dfbef7acd48233253616ec60b154c87bc92955646e0ccae`

Status:

`PAYOFF_MODEL_FAILED_REPLICATION`

Motivos:

- `no_one_position_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## Melhor candidato encontrado

- target: 1 bps;
- trailing distance: 2 bps;
- modelo: `ridge_payoff`;
- top-p: 2%;
- sinais selecionados: 82;
- gross médio selecionado: +1,0657 bps;
- features: 122.

## Resultado agregado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Janelas positivas em 1 bps | Pior janela em 1 bps | Janelas com min exec. | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 82 | +1,0978 | +0,0978 | -0,9022 | 3/6 | -1,9312 | 6/6 | 17 BREAKEVEN_STOP / 49 TIMEOUT / 16 TRAILING_STOP |
| uma posição | 57 | +1,8744 | +0,8744 | -0,1256 | 4/6 | -0,9928 | 6/6 | 11 BREAKEVEN_STOP / 32 TIMEOUT / 14 TRAILING_STOP |

## Por janela

| Janela | Sinais | One-position exec. | One-position gross | One-position net 1 bps | One-position net 2 bps | Observação |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fast24 | 14 | 10 | +2,5927 | +1,5927 | +0,5927 | passa |
| Fast28 | 14 | 9 | +0,6506 | -0,3494 | -1,3494 | falha custo 1 bps |
| Fast29 | 14 | 10 | +2,2210 | +1,2210 | +0,2210 | passa |
| Fast32 | 13 | 9 | +0,0072 | -0,9928 | -1,9928 | falha |
| Fast35 | 14 | 13 | +3,4639 | +2,4639 | +1,4639 | passa |
| Fast36 | 13 | 6 | +1,2918 | +0,2918 | -0,7082 | passa em 1 bps, falha em 2 bps |

## Decisão

Não temos estratégia.

O scout curto melhorou a aderência à ideia operacional desejada — target pequeno,
trailing rápido, top-p baixo e uma posição por vez — mas ainda não passou na
barra mínima:

- não replicou em todas as janelas com 1 bps de custo;
- ficou negativo no agregado com 2 bps de custo;
- Fast28 e Fast32 quebraram mesmo no modo one-position;
- o resultado independente ficou quase zerado em 1 bps e negativo em stress;
- a maior parte das saídas ainda foi TIMEOUT, não trailing capturando movimento.

Conclusão:

A hipótese "trades bem curtos resolvem o problema" foi testada e rejeitada
nesta amostra. O que existe é sinal fraco/regime-dependent, não uma base sólida
para promover a Testnet.

Sem Testnet/LIVE.
