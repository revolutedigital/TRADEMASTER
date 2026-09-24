# Fast105a–Fast105c — encerramento da hipótese do veto fixo

Resposta objetiva: a hipótese Fast105 não virou estratégia. Com três janelas
prospectivas, o agregado ficou negativo após custo e só uma janela foi positiva.

Essa linha testava a melhor pista do Fast104:

```text
hist_gradient_payoff + target 10 bps + trailing 15 bps + top 5%
+ veto fixo spot_return_1s_bps >= 0
```

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

## Qualidade das janelas

| Janela | Observação de qualidade |
|---|---|
| fast105a | Captura normal; `588` decisões pós-pré-registro |
| fast105b | Captura degradada por reconnects; só `160` decisões pós-pré-registro |
| fast105c | Captura normal; `586` decisões pós-pré-registro |

Fast105b foi positiva, mas é a janela menos confiável pela degradação de
stream. Fast105a e Fast105c, que são mais limpas e com contagem parecida, foram
negativas.

## Resultado por janela

| Janela | Decisões pós-pré-registro | Top 5% antes do veto | Após veto | Trades one-position | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|---:|---:|
| fast105a | `588` | `30` | `21` | `4` | `-10.019151404020977` | `-14.019151404020977` |
| fast105b | `160` | `8` | `7` | `2` | `+10.524744521861907` | `+8.524744521861907` |
| fast105c | `586` | `30` | `28` | `2` | `-4.989880917747832` | `-6.989880917747832` |

Exit reasons:

- fast105a: `TIMEOUT: 4`;
- fast105b: `TIMEOUT: 2`;
- fast105c: `TIMEOUT: 2`.

## Resultado agregado A+B+C

Status do runner:

`FAST105_INSUFFICIENT_PROSPECTIVE_EVIDENCE`

Resumo:

| Métrica | Valor |
|---|---:|
| Janelas elegíveis | `3` |
| Trades one-position | `8` |
| Sinais após veto | `56` |
| Gross total | `+3.5157122000930965` bps |
| Net @ 1 bps | `-4.4842877999069035` bps |
| Net @ 2 bps | `-12.484287799906904` bps |
| Profit médio @ 1 bps | `-0.5605359749883629` bps/trade |
| Stress médio @ 2 bps | `-1.560535974988363` bps/trade |
| Fração de janelas positivas em stress | `1/3` |

Gates:

| Gate | Resultado |
|---|---|
| Mínimo `30` trades | Falhou (`8`) |
| Mínimo `3` janelas | Passou (`3`) |
| Stress médio > `+1.5 bps/trade` | Falhou (`-1.5605`) |
| Expected médio > `+2.5 bps/trade` | Falhou (`-0.5605`) |
| Fração de janelas positivas em stress >= `75%` | Falhou (`33.3%`) |
| Falha precoce total formal | Não disparou por trade count (`8 < 10`) |
| Falha precoce por janela formal | Não disparou |

## Decisão

Encerrar esta hipótese como não estratégia.

Ela não atingiu o gatilho formal de falha precoce porque acumulou só `8` trades,
mas o conjunto já é suficiente para uma decisão de pesquisa:

- duas janelas limpas negativas;
- uma janela positiva degradada;
- agregado negativo em custo esperado e stress;
- fração de janelas positivas muito baixa;
- custo ainda destrói a margem.

Não vale gastar mais coleta nesta família sem uma mudança de hipótese. E uma
mudança de hipótese exigiria novo pré-registro.

## O que isso nos ensina

O veto `spot_return_1s_bps >= 0` parecia bom no diagnóstico Fast104, mas não
sobreviveu bem quando congelado e levado para frente. Isso reforça que:

- trailing stop não corrige entrada ruim;
- filtro de micro-regime precisa ser mais estrutural que um retorno de 1 segundo;
- o custo continua sendo o inimigo principal;
- edge diagnóstico em janelas pequenas vira miragem facilmente.

## Próximo caminho recomendado

Parar Fast105 e pivotar para outra família.

Se continuar a busca, a próxima família não deve ser mais uma variação desse
mesmo veto. O próximo pré-registro deveria procurar uma destas rotas:

1. outro mercado/par com menor ruído e custo proporcional menor;
2. timeframe maior, com alvo maior e menos dependência de microsegundo;
3. modelo de regime separado antes do modelo de entrada, em vez de veto simples
   pós-score.

## Arquivos e hashes

Fast105c arquivado:

- report:
  `backend/data/microstructure_v1/reports/fast105c-prospective-observation/report.json`;
- report sha256:
  `fa8ede0847e2b27e8cd626d203ca92591946c39b87835b700b3c61d5db90d2ab`;
- one-position settlements sha256:
  `c30ef4dde14e8290c1cdae5029e63d02a3a87a169c5ca0b92c1592705bd35c69`.

Fast105a+b+c arquivado:

- report:
  `backend/data/microstructure_v1/reports/fast105a-fast105c-prospective-observation/report.json`;
- report sha256:
  `2f25e90663dda6522b731e10055840b7298970db622f26502657282a810cc2a8`;
- scored decisions sha256:
  `38484d6c85a38e7a14854e2b7029ccf758f5b9546d012084e110451d45c7cf72`;
- selected after fixed veto sha256:
  `8a757991a027d87f3a678817d93e1f24d3b494ad8788219a5cdeac1dde4780b0`;
- one-position settlements sha256:
  `a3ecc0bd84aa674a91282bcbf28883c6fab392e8dc3c5f77acaa13bc3d6c5b67`.
