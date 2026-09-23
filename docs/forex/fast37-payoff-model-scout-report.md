# Fast37 — scout de modelo por payoff executável

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32, Fast35, Fast36  
Validação: leave-one-window-out  
Resultado: modelo de payoff limpo também falhou replicação por janela.

## Por que esta rodada existe

Fast36 mostrou que modelar "probabilidade de tocar alvo" não basta. O candidato
anterior podia escolher sinais com target-rate aparentemente alto e ainda perder
dinheiro quando incluímos stop, spread, overlap e custo.

Fast37 muda o rótulo de treino:

- antes: `paid_expected_before_stop` / target touch;
- agora: `gross_executable_bps` de replay top-of-book com trailing stop.

O objetivo é ranquear sinais pelo payoff executável previsto, não pela chance
de tocar um alvo isolado.

## Leakage guard

Antes do resultado final, o script foi corrigido para excluir colunas de replay
e futuro da matriz de features:

- `gross_executable_bps`;
- `maximum_favorable_executable_bps`;
- `maximum_adverse_executable_bps`;
- entry/exit time/price;
- exit reason;
- `model_score`;
- colunas com sufixo `_replay`;
- labels/futuros já bloqueados por `LABEL_AND_FUTURE_COLUMNS`.

Teste unitário:

`backend/tests/unit/test_book_payoff_model_scout.py`

## Fonte

Report:

`backend/data/microstructure_v1/reports/book-payoff-model-scout-fast24-fast28-fast29-fast32-fast35-fast36-100ms/report.json`

Hash:

`e3dac41ce464c8c6458b652544fba38d4370a0b1ea965e7a5e985a5b5ca0af81`

Status:

`PAYOFF_MODEL_FAILED_REPLICATION`

Motivos:

- `no_one_position_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## Grade testada

- target / ativação: 3, 5 e 8 bps;
- trailing distance: 3, 5 e 8 bps;
- top-p: 2%, 5%, 10% e 20%;
- modelos:
  - `hist_gradient_payoff`;
  - `ridge_payoff`;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- minimum executable per window: 5.

Total: 72 candidatos.

## Melhor candidato limpo

| Campo | Valor |
| --- | --- |
| Modelo | `hist_gradient_payoff` |
| Top-p | 2% |
| Target / ativação | 8 bps |
| Trailing distance | 8 bps |
| Feature count | 122 |
| Sinais selecionados | 81 |
| Média selecionada bruta | +2,1614 bps |

### Agregado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Janelas positivas em 1 bps | Pior janela em 1 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 81 | +2,0472 | +1,0472 | +0,0472 | 4/6 | -5,4088 | 5 INITIAL_STOP / 66 TIMEOUT / 10 TRAILING_STOP |
| uma posição | 22 | +2,3961 | +1,3961 | +0,3961 | 3/6 | -2,7393 | 1 INITIAL_STOP / 16 TIMEOUT / 5 TRAILING_STOP |

### Por janela — melhor candidato

| Janela | Modo | Exec. | Gross | Net 1 bps | Net 2 bps | Saídas |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Fast24 | independente | 14 | +1,9515 | +0,9515 | -0,0485 | 13 TIMEOUT / 1 TRAILING_STOP |
| Fast24 | uma posição | 4 | +1,6184 | +0,6184 | -0,3816 | 3 TIMEOUT / 1 TRAILING_STOP |
| Fast28 | independente | 14 | +4,2552 | +3,2552 | +2,2552 | 14 TIMEOUT |
| Fast28 | uma posição | 1 | +3,8629 | +2,8629 | +1,8629 | 1 TIMEOUT |
| Fast29 | independente | 14 | +0,7592 | -0,2408 | -1,2408 | 14 TIMEOUT |
| Fast29 | uma posição | 2 | +0,4906 | -0,5094 | -1,5094 | 2 TIMEOUT |
| Fast32 | independente | 13 | +5,0872 | +4,0872 | +3,0872 | 13 TIMEOUT |
| Fast32 | uma posição | 2 | +0,1168 | -0,8832 | -1,8832 | 2 TIMEOUT |
| Fast35 | independente | 14 | -4,4088 | -5,4088 | -6,4088 | 4 INITIAL_STOP / 6 TIMEOUT / 4 TRAILING_STOP |
| Fast35 | uma posição | 6 | -1,7393 | -2,7393 | -3,7393 | 4 TIMEOUT / 2 TRAILING_STOP |
| Fast36 | independente | 12 | +5,3240 | +4,3240 | +3,3240 | 1 INITIAL_STOP / 6 TIMEOUT / 5 TRAILING_STOP |
| Fast36 | uma posição | 7 | +7,3713 | +6,3713 | +5,3713 | 1 INITIAL_STOP / 4 TIMEOUT / 2 TRAILING_STOP |

## Decisão

Não temos estratégia.

Mesmo treinando diretamente em payoff executável:

- o agregado parece positivo;
- mas a consistência por janela falha;
- Fast35 quebra a tese com perda relevante;
- em one-position, só 3/6 janelas ficam positivas a 1 bps;
- várias janelas têm menos de 5 execuções no modo one-position.

Conclusão:

O problema não era só o rótulo "tocou alvo". A família atual de features/modelos
ainda não é estável o suficiente para operar.

Próximo caminho recomendado:

1. abandonar a família Fast34-Fast37 como estratégia candidata;
2. mudar para busca de regime/filtro de estabilidade antes de entrada;
3. tratar Fast35 como regime explicitamente adverso e tentar aprender um
   "no-trade gate" prospectivo, não otimizar trailing.

Sem Testnet/LIVE.
