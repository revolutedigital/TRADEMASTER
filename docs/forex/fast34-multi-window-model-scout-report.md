# Rodada 34 — modelo composto multi-janela com trailing stop

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32  
Validação: leave-one-window-out  
Latência: 100ms  
Saída simulada: stop inicial, breakeven no alvo de ativação e trailing stop  
Resultado: ainda não temos estratégia aprovada.

## Fonte

Report:

`backend/data/microstructure_v1/reports/book-multi-window-model-scout-fast24-fast28-fast29-fast32-100ms/report.json`

Hash:

`ffc6722eb640cf6b9fa9bea3b363f15f1e3237fbcf96ef9222690c93711d3413`

Status:

`MULTI_WINDOW_MODEL_FAILED_REPLICATION`

Motivos:

- `no_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## O que foi testado

O scout trocou a regra fixa por modelos que rankeiam entradas usando apenas
features causais conhecidas no momento da decisão:

- fluxo e retorno curtíssimo;
- pressão, imbalance e microprice do book;
- basis spot/perp e idade do spot;
- mark/index/funding quando disponível;
- lado da operação codificado sem olhar o futuro.

Grade testada:

- alvo de ativação/breakeven: 3, 5 e 8 bps;
- trailing distance: 3, 5 e 8 bps;
- top-p: 2%, 5%, 10% e 20%;
- modelos: `logistic_balanced` e `hist_gradient`;
- custo explícito: 0, 0,25, 0,5, 1 e 2 bps;
- critério: treina em 3 janelas e segura a janela restante fora da amostra.

Total: 72 candidatos.

## Melhor candidato encontrado

| Campo | Valor |
| --- | --- |
| Modelo | `hist_gradient` |
| Top-p | 5% |
| Alvo / breakeven | 8 bps |
| Trailing distance | 3 bps |
| Sinais selecionados | 132 |
| Target-rate médio selecionado | 13,64% |
| Feature count | 122 |

### Agregado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Janelas positivas em 1 bps | Pior janela em 1 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 132 | +2,1485 | +1,1485 | +0,1485 | 2/4 | -1,2686 | 126 TIMEOUT / 6 TRAILING_STOP |
| uma posição | 24 | +3,6241 | +2,6241 | +1,6241 | 4/4 | +0,6793 | 21 TIMEOUT / 3 TRAILING_STOP |

### Por janela — melhor candidato

| Janela | Modo | Exec. | Gross | Net 1 bps | Net 2 bps | Saídas |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Fast24 | independente | 33 | -0,2686 | -1,2686 | -2,2686 | 32 TIMEOUT / 1 TRAILING_STOP |
| Fast24 | uma posição | 10 | +4,5849 | +3,5849 | +2,5849 | 9 TIMEOUT / 1 TRAILING_STOP |
| Fast28 | independente | 33 | +3,9724 | +2,9724 | +1,9724 | 29 TIMEOUT / 4 TRAILING_STOP |
| Fast28 | uma posição | 6 | +3,9979 | +2,9979 | +1,9979 | 4 TIMEOUT / 2 TRAILING_STOP |
| Fast29 | independente | 33 | +0,7457 | -0,2543 | -1,2543 | 32 TIMEOUT / 1 TRAILING_STOP |
| Fast29 | uma posição | 6 | +1,6793 | +0,6793 | -0,3207 | 6 TIMEOUT |
| Fast32 | independente | 33 | +4,1443 | +3,1443 | +2,1443 | 33 TIMEOUT |
| Fast32 | uma posição | 2 | +3,5334 | +2,5334 | +1,5334 | 2 TIMEOUT |

## Decisão

Não temos estratégia.

O achado útil é que o modo "uma posição por vez" começou a mostrar algo que
parece mais próximo de um edge: todas as quatro janelas ficaram positivas em
1 bps no melhor candidato. Mas isso ainda não basta porque:

- o modo independente falha em 2 de 4 janelas no custo de 1 bps;
- o modo uma-posição falha no estresse de 2 bps em Fast29;
- Fast32 teve só 2 execuções no modo uma-posição, abaixo do mínimo configurado;
- a maior parte das saídas é TIMEOUT, não trailing stop, então o edge parece
  mais dependente de seleção/holding do que de captura ativa de rompimento;
- o próprio status do report é reprovação de replicação.

Conclusão: existe um candidato de pesquisa para nova rodada shadow, mas não uma
estratégia operável. Sem Testnet/LIVE.

## Próximo caminho

1. Não tentar promover Fast34 para estratégia.
2. Separar uma tese específica: "hist_gradient top 5%, alvo 8 bps, trailing 3 bps,
   uma posição por vez".
3. Rodar uma janela futura prospectiva com esse candidato congelado, sem mexer
   nos parâmetros.
4. Só discutir Testnet se a janela futura mantiver:
   - todas as janelas positivas em 1 bps no modo uma-posição;
   - mínimo de execuções por janela;
   - drawdown aceitável;
   - custo de 2 bps pelo menos não catastrófico.
