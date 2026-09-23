# Fast53 — holdout estendido do candidato low-cost micro

Data: 2026-09-23  
Mercado: BTCUSDT futures, `aggTrades` normalizado  
Modelo congelado: Fast43, treinado/calibrado apenas em 2026-06-30 a 2026-07-06  
Holdout total: 2026-07-07 a 2026-09-20  
Resultado: falhou como estratégia robusta.

## Candidato testado

Mesmo artefato de Fast43–45:

- model hash: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- alvo: `latency_paid_stress_before_stop`;
- horizonte: 300 s;
- features: `flow_price_session`;
- top-p: 1%;
- threshold congelado: `0,90195311025804`;
- política: `micro_8_3`;
- custo esperado: 1 bps roundtrip;
- custo stress: 2 bps roundtrip;
- latência: 100 ms;
- uma posição por vez.

Não houve retreino nos holdouts.

## Resultado por bloco

| Rodada | Holdout | Status | Sinais | Trades | Média 1 bps | Média 2 bps | Total 2 bps | Max DD |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fast43 | 2026-07-07 a 2026-07-13 | stress positivo | 432 | 100 | +2,0319 | +1,0319 | +103,1949 | 83,2690 |
| Fast44 | 2026-07-14 a 2026-07-20 | stress positivo | 369 | 89 | +3,1991 | +2,1991 | +195,7242 | 54,2140 |
| Fast45 | 2026-07-21 a 2026-07-27 | stress positivo | 120 | 25 | +8,2941 | +7,2941 | +182,3520 | 18,6419 |
| Fast46 | 2026-07-28 a 2026-08-03 | esperado positivo / stress negativo | 335 | 80 | +0,7604 | -0,2396 | -19,1699 | 90,9618 |
| Fast47 | 2026-08-04 a 2026-08-10 | stress positivo | 66 | 17 | +6,0499 | +5,0499 | +85,8490 | 17,0089 |
| Fast48 | 2026-08-11 a 2026-08-17 | stress positivo | 70 | 16 | +2,5789 | +1,5789 | +25,2629 | 34,0240 |
| Fast49 | 2026-08-18 a 2026-08-24 | falhou | 4.176 | 920 | -0,0170 | -1,0170 | -935,6237 | 417,2261 |
| Fast50 | 2026-08-25 a 2026-08-31 | esperado positivo / stress negativo | 938 | 220 | +0,0434 | -0,9566 | -210,4577 | 148,2068 |
| Fast51 | 2026-09-01 a 2026-09-07 | falhou | 509 | 120 | -0,1729 | -1,1729 | -140,7492 | 165,8278 |
| Fast52 | 2026-09-08 a 2026-09-14 | falhou | 785 | 172 | -1,2957 | -2,2957 | -394,8531 | 344,4018 |
| Fast53 | 2026-09-15 a 2026-09-20 | falhou | 1.453 | 299 | -1,3971 | -2,3971 | -716,7299 | 529,2585 |

## Consolidado

| Métrica | Valor |
| --- | ---: |
| Blocos positivos em stress | 5/11 |
| Blocos negativos em stress | 6/11 |
| Trades | 2.058 |
| Dias com trades | 68 |
| Gross médio | +1,1131 bps |
| Net médio com 1 bps | +0,1131 bps |
| Net médio com 2 bps | -0,8869 bps |
| Total com 1 bps | +232,7995 bps |
| Total com 2 bps | -1.825,2005 bps |
| Win rate com 1 bps | 66,28% |
| Win rate com 2 bps | 44,95% |
| Dias positivos em stress | 35/68 |
| Pior dia em stress | -383,3756 bps |
| Melhor dia em stress | +132,6866 bps |
| t-stat por total diário em stress | -2,22 |

Saídas:

- `BREAKEVEN_STOP`: 1.277;
- `INITIAL_STOP`: 670;
- `TRAILING_STOP`: 81;
- `TIMEOUT`: 30.

## Leitura

Encontramos um edge de regime, não uma estratégia.

O candidato foi forte de 2026-07-07 a 2026-07-27, sobreviveu parcialmente até
2026-08-17, e quebrou de forma clara a partir de 2026-08-18. A maior perda veio
quando o threshold passou a selecionar muito mais sinais:

- Fast48: 70 sinais, 16 trades, stress positivo;
- Fast49: 4.176 sinais, 920 trades, stress fortemente negativo;
- Fast53: 1.453 sinais, 299 trades, stress negativo.

Isso sugere que o probability score continuou detectando "anda pelo menos 2 bps
antes do stop", mas esse evento deixou de ser suficiente para pagar a política
micro depois de custo. O problema virou regime/excesso de exposição, não apenas
entrada individual.

## Decisão

Não temos estratégia aprovada.

Temos um candidato/regime que merece uma rodada nova:

1. gate de regime/no-trade antes do top-p;
2. limitador de densidade de sinais, porque os blocos ruins explodiram a
   quantidade de entradas;
3. retestar com o artefato congelado + gate escolhido sem olhar o bloco de
   teste;
4. só promover se sobreviver por mês e stress de 2 bps.

Sem Testnet/LIVE.

## Hashes dos blocos novos

| Rodada | Report | Trades |
| --- | --- | --- |
| Fast46 | `fcdeb70ca0cfdc4c91c0ea2dbf21986d3fe7d9aa09ec0ade791f42d0c1bd738b` | `77e73d1db8e53886545cbe24af0aec5cb51b45463d0460b3dc4a4c2561e1366f` |
| Fast47 | `d0f4364539c3d7b6e66ad1a87a8ab1b7e415967507d3fe6a4038ab98c31b3733` | `6028e77b335ab77cb4676affac8abf6eefa64f2a7f6c70c53786469a88065c45` |
| Fast48 | `6120fb358717f66efce47ec51a8ddcf39f12bb5644a186f982cbf8549c8884fa` | `d258551f6821ccc1d97621c4c706a40dfb2e1e4462e9a17d70ff7e89376ee187` |
| Fast49 | `fc95ae15a40c661bdbae6a3465df7fc90f2ef77ae347aca2018752109776a5ac` | `d91ea21b36d32718813cfbfa142284bda4aaa16c3c4deb37b05667df35d4e322` |
| Fast50 | `a95ddcb928c6e0f1f11e0d8e289a7d872134d01a42b5ac46169c8335409bce35` | `9f84deff2fc2fb90fcf6f7fe847ee112b0fe8ded7b81f1da2779c13bf5959f18` |
| Fast51 | `cbcc34d494095c8244dc9f3caee32fd59778b80f8bfd793888ab32978cef2121` | `552bb2333cafa6fa7b181033f653f4d04b4f640ba15efdcabfbea89658bc637c` |
| Fast52 | `f9fddc1563939fdafd48a0f7f629baa7e686250418da90eb4c4678814859dd7d` | `7ed7fe184de02cde8922e2bbd569b87bd2e889e06bd9f52032e06f0c90f6a008` |
| Fast53 | `b8207e71f840602c818f151ff99954b427dffc09ea8cbc597c0138ed97bfbbd1` | `3dde3b14c5f6c68ebac4ac16ddafffe643088b445b816bc3aa0158a635d1b544` |
