# Rodada 30 — agregação das janelas shadow Fast24 + Fast28 + Fast29

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29  
Latência agregada: 100ms  
Resultado: a candidata não passa como estratégia executável. Ela sobrevive no modo independente em 1 bps, mas falha no modo uma posição por vez e em 2 bps.

## Ferramenta adicionada

`backend/scripts/research/summarize_book_fixed_gate_shadow_windows.py`

O script:

- lê settlement reports já emitidos;
- valida `research_only=true`;
- valida `order_submission_allowed=false`;
- valida `execution_authorization=none`;
- valida `approved_strategy=false`;
- agrega por modo de execução;
- calcula média ponderada por trades executáveis;
- exige consistência por janela, não só média agregada.

## Fontes

| Janela | Settlement 100ms | Observação |
| --- | --- | --- |
| Fast24 | `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-settlement-fast24-dry-run-100ms/report.json` | positiva em 1 bps; negativa em 2 bps |
| Fast28 | `backend/data/microstructure_v1/reports/book-fixed-gate-prospective-window-fast28-30m-100ms/settlement-100ms/report.json` | positiva em 1 bps; negativa em 2 bps |
| Fast29 | `backend/data/microstructure_v1/reports/book-fixed-gate-prospective-window-fast29-30m-100ms/settlement-100ms/report.json` | independente positiva em 1 bps; uma posição negativa |

Summary report:

- path: `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-window-summary-fast24-fast28-fast29-100ms/report.json`;
- hash: `5987078e9babb48ee30b73b85764e7491d9217d4681eca1c54ad79963e0650ff`.

## Resultado agregado

Status:

`SHADOW_WINDOW_SUMMARY_FAILED_PROFIT_COST_REPLICATION`

Motivos:

- `independent_has_non_positive_window_at_2bps`;
- `one_position_has_non_positive_window_at_1bps`;
- `one_position_has_non_positive_window_at_2bps`.

| Modo | Janelas positivas em 1 bps | Trades exec. | Net médio 1 bps | Net médio 2 bps | Pior janela em 1 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 3/3 | 95 | +0,2879 | -0,7121 | +0,1173 | 36 TARGET / 59 TIMEOUT |
| uma posição | 2/3 | 31 | +0,3029 | -0,6971 | -1,6421 | 12 TARGET / 19 TIMEOUT |

## Decisão

A candidata `spot_age_fixed_gate_v1` não está aprovada.

O melhor resumo é:

- existe sinal estatístico fraco no modo independente;
- o sinal é pequeno e depende de custo total perto de 1 bps;
- em 2 bps morre;
- em execução sequencial realista falhou em Fast29;
- portanto, não é bot pronto nem candidata para Testnet.

O que aprendemos:

- o gate de frescor de spot continua útil como feature/regime;
- operar todos os sinais sobrepostos melhora estatística, mas é menos realista operacionalmente;
- restringir para uma posição por vez expõe instabilidade;
- próximo caminho deve ser buscar uma política que use esse gate como filtro, mas selecione melhor quais sinais carregar.

Próximo passo recomendado:

1. manter o gate de `spot_update_age_ms` como feature;
2. procurar um seletor adicional para diferenciar os 7 sinais fracos do Fast29 dos sinais bons Fast24/Fast28;
3. testar gestão de saída alternativa somente se o seletor de entrada melhorar, não usar trailing stop para mascarar entrada ruim;
4. continuar sem Testnet/LIVE até nova política passar shadow prospectivo.
