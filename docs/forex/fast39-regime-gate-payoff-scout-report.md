# Fast39 — scout nested de regime/no-trade gate

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32, Fast35, Fast36  
Tese: aplicar um gate de regime antes do modelo de payoff executável  
Resultado: falhou replicação.

## Por que esta rodada existe

Fast37 mostrou que o modelo de payoff limpo ainda quebra por regime. Fast38
indicou que Fast35 e Fast36 têm drift de microestrutura/volume/fluxo. Fast39
testa se um gate simples de no-trade, escolhido sem olhar o holdout, consegue
bloquear regimes ruins antes da entrada.

## Método

Para cada janela de holdout:

1. remove a janela de teste;
2. nas janelas restantes, gera previsões out-of-fold do modelo de payoff;
3. escolhe um gate de uma feature causal por threshold;
4. congela esse gate;
5. treina o modelo final nas janelas de treino;
6. aplica gate + modelo na janela segurada fora.

Isso evita escolher o gate olhando o resultado da janela de teste.

Configuração:

- modelo: `hist_gradient_payoff`;
- target / ativação: 8 bps;
- trailing distance: 8 bps;
- top-p: 2%;
- gate keep fractions: 50%, 70%, 85%, 100%;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- mínimo por janela: 3 execuções one-position.

## Fonte

Report:

`backend/data/microstructure_v1/reports/book-regime-gate-payoff-scout-fast24-fast28-fast29-fast32-fast35-fast36-100ms/report.json`

Hash:

`539bd5058e869b4b79d75928d6220321345bd2a6e3a0aa8ba0359bb706ea6ea4`

Status:

`REGIME_GATE_PAYOFF_FAILED_REPLICATION`

Motivos:

- `no_one_position_gate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

## Resultado agregado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Janelas positivas em 1 bps | Pior janela em 1 bps | Janelas com min exec. | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 81 | +1,4771 | +0,4771 | -0,5229 | 4/6 | -5,4088 | 6/6 | 5 INITIAL_STOP / 66 TIMEOUT / 10 TRAILING_STOP |
| uma posição | 25 | +2,1410 | +1,1410 | +0,1410 | 4/6 | -3,7515 | 4/6 | 19 TIMEOUT / 6 TRAILING_STOP |

## Por janela

| Janela | Gate escolhido | One-position exec. | One-position net 1 bps | One-position net 2 bps | Observação |
| --- | --- | ---: | ---: | ---: | --- |
| Fast24 | `directed_mark_index_basis_bps <= -0,007409` | 5 | +0,3185 | -0,6815 | passa 1 bps, falha 2 bps |
| Fast28 | `hour_cos <= -0,848241` | 1 | +2,8629 | +1,8629 | lucro, mas amostra insuficiente |
| Fast29 | `directed_mark_index_basis_bps <= -0,007409` | 4 | -3,7515 | -4,7515 | falha |
| Fast32 | `book_bid_replenishment_qty_30s >= 140,5175` | 1 | +0,6026 | -0,3974 | amostra insuficiente |
| Fast35 | `hour_sin <= -0,005272` | 6 | -2,7393 | -3,7393 | falha |
| Fast36 | `directed_book_depth_imbalance_change_5s <= 0` | 8 | +6,8634 | +5,8634 | passa |

## Decisão

Não temos estratégia.

O gate de regime simples não resolveu:

- Fast35 continuou negativo;
- Fast29 também ficou negativo;
- Fast28/Fast32 ficaram com amostra one-position insuficiente;
- o agregado continuou positivo, mas sem consistência por janela;
- os gates escolhidos no treino já tinham sinais de fragilidade, com mínimos
  negativos em vários folds.

Conclusão:

O problema não é só "falta um filtro simples". A família atual precisa de uma
mudança mais estrutural:

- mais janelas para modelar regime de forma estatisticamente útil; ou
- outra família de sinal com edge mais direto; ou
- mudar o mercado/horizonte/custo, porque no setup atual os sinais ficam
  regime-dependent demais.

Sem Testnet/LIVE.
