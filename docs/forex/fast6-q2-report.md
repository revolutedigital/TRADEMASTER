# Rodada 6 — resultado Q2 da fila global Top P

Status: **reprovada em 2021-Q2**. Nenhuma política foi congelada para validação H2. 2021-H2,
2022, D4, S3 e a semana da corretora continuam fechados.

## Calibração da entrada

- 4.739.097 eventos em 30 partições; 4.505.973 labels válidos e hashes conferidos;
- 1.255.538 previsões direcionais de referência;
- 40/40 modelos local/lado/cenário melhoraram o Brier da taxa-base na auditoria;
- melhora relativa mediana de Brier: 4,93%; mínima: 1,39%;
- calibradores escolhidos: 17 Platt, 15 beta e 8 isotônicos;
- blend escolhido: 19 locais puros, 9 globais puros e 12 híbridos;
- seis pisos Top P distintos, de 37,15% no Top 0,25% a 25,20% no Top 10%.

Isso confirma que o sistema consegue produzir e ordenar probabilidades mais informativas que a
taxa-base. Não confirma expectativa econômica positiva.

## Resultado das 18 políticas

| Top P | Melhor trailing | Trades | Média base | Média stress | PF stress | Meses positivos |
|---:|---:|---:|---:|---:|---:|---:|
| 0,25% | 0,5R | 290 | -0,241R | -0,446R | 0,269 | 0/3 |
| 0,50% | 0,5R | 459 | -0,258R | -0,450R | 0,274 | 0/3 |
| 1,00% | 0,5R | 683 | -0,274R | -0,462R | 0,258 | 0/3 |
| 2,00% | 1,5R | 905 | -0,259R | -0,472R | 0,237 | 0/3 |
| 5,00% | 1,5R | 1.570 | -0,244R | -0,454R | 0,240 | 0/3 |
| 10,00% | 1,5R | 2.792 | -0,263R | -0,458R | 0,232 | 0/3 |

Todas as 18 tentativas tiveram média negativa em base e estresse, bootstrap inferior negativo e
nenhum mês positivo. Portanto a reprovação independe dos gates de diversidade ou concentração.

## Diagnóstico da melhor tentativa

Top 0,25%, trailing 0,5R:

| Cenário | Ativação | Média quando ativou | Média sem ativar | Stops -1R | Timeouts | Runners |
|---|---:|---:|---:|---:|---:|---:|
| Base | 40,69% | +0,571R | -0,798R | 126 | 46 | 118 |
| Stress | 29,31% | +0,544R | -0,856R | 162 | 43 | 85 |

No base, a taxa de ativação necessária para empatar essa distribuição seria aproximadamente 58%;
no stress, 61%. O ranking melhorou a chance de o preço andar, mas ainda deixou stops demais antes
de +0,5R. Os runners existem — máximo +3,66R base e +3,45R stress — porém não pagaram os stops.

## Decisão

Não abrir H2 e não ativar estratégia. A próxima hipótese deve tratar o gargalo observado sem mudar
a entrada olhando H2: mover a proteção antes de +0,5R e testar timeouts pré-ativação menores. Isso
exige nova rodada pré-registrada e conta como novas tentativas.

Artefato Q2: `aca93b63bf23462ce09e5465ced2bbaf43ef181233dab182839a04a9c976afcd`.
