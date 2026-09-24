# Fast125 — USDJPY compression breakout top-p model audit

Fast125 testou a tese: "dá para usar um modelo de probabilidade/top-p para
escolher só as melhores entradas da regra USDJPY H1 compression breakout?"

A resposta do audit histórico walk-forward foi: **não o suficiente para virar
estratégia**.

Status:

`FAST125_USDJPY_COMPRESSION_TOPP_MODEL_FAILED_OOS`

## Protocolo

Fonte:

- `backend/data/lab_fast122/usdjpy-compression-breakout-temporal-audit/frozen-policy-trades.parquet`;
- regra congelada: `tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r`.

Para cada ano de teste entre `2016` e `2026`:

1. treina somente com anos anteriores;
2. calcula score de entrada usando features conhecidas no momento da entrada;
3. escolhe modelo/top-p usando apenas o treino;
4. aplica o threshold no ano seguinte.

Modelos testados:

- `ridge_expected_pips`: regressão linear regularizada para expectancy em
  `net_stress_pips`;
- `logistic_positive`: probabilidade de `net_stress_pips > 0`.

Top-p testado:

- `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, `0.75`, `1.00`.

Features de entrada:

- lado;
- distância do breakout;
- ATR;
- risco em pips;
- compressão;
- threshold de compressão;
- razão compressão/threshold;
- breakout/ATR;
- hora, weekday e mês com encoding cíclico.

## Resultado agregado OOS

Comparação no período walk-forward `2016–2026`:

| Métrica | Regra crua | Modelo top-p |
|---|---:|---:|
| trades | `1,426` | `333` |
| média stress pips/trade | `1.9559` | `2.1619` |
| LCB 95% stress pips/trade | `0.3785` | `-0.5689` |
| total stress pips | `2,789.10` | `719.93` |
| anos positivos | `6/11` | `9/11` |
| menor nº trades/ano | `44` | `4` |
| pior ano médio | `-3.7258` | `-6.3327` |
| lift vs regra crua | — | `+0.2061` |

O top-p melhora levemente a média e a taxa de anos positivos, mas:

- perde significância estatística (`LCB` fica negativo);
- gera anos esparsos demais (`4` trades em 2025, `8` em 2021);
- tem pior ano mais feio que a regra crua (`2019 = -6.33` pips/trade);
- entrega lift muito pequeno versus simplesmente operar a regra crua.

## Status reasons

- `selected_stress_lower_confidence_bound_not_positive`;
- `stress_lift_vs_unfiltered_baseline_too_small`;
- `yearly_trade_count_too_sparse`;
- `worst_year_drawdown_too_large`.

## Interpretação

O modelo top-p não salvou a tese. Ele parece selecionar alguns trades muito
bons, mas não com estabilidade suficiente. O resultado também mostra uma coisa
importante: a regra crua, no recorte `2016–2026`, já é mais estatisticamente
limpa que o top-p, porque mantém LCB positivo e amostra maior.

Isso não aprova a regra crua, porque o Fast122 já mostrou fragilidade temporal
no período anterior. Mas derruba a hipótese de que "só um modelo escolhendo as
entradas" resolveria o problema.

## Segurança

Fast125 permanece pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast125_usdjpy_compression_topp_model.py`;
- teste:
  `backend/tests/unit/test_fast125_usdjpy_compression_topp_model.py`;
- report JSON:
  `backend/data/lab_fast125/usdjpy-compression-topp-model/report.json`;
- candidatos por fold:
  `backend/data/lab_fast125/usdjpy-compression-topp-model/fold-candidates.csv`;
- trades OOS selecionados:
  `backend/data/lab_fast125/usdjpy-compression-topp-model/selected-oos-trades.parquet`.

Hashes:

- report:
  `6187ca7e532706509d00c8824f8bf18b63373112064490b622624908d400e4ea`;
- fold candidates:
  `97de560fb51bc477cd48f7972ebadb28484acefa5c26c10d0c7b24c9aab54f54`;
- selected OOS trades:
  `34277a95800d41fcec247deb6898b811c8743375d5677611b3104d4edda38c96`.

## Próxima decisão técnica

Fast124 deixou a melhor hipótese congelada para dados futuros. Fast125 mostrou
que top-p/modelo simples não melhora o bastante.

O próximo caminho com maior chance não é mais mexer em modelo sobre essa mesma
entrada. É procurar uma família de sinal diferente ou usar a regra crua apenas
como observação prospectiva até entrar dado novo pós-pré-registro.
