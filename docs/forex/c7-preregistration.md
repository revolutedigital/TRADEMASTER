# Pré-registro C7: o Claude como banca, nas decisões diárias do último ano

Registrado em 2026-09-25, depois do relatório de C6 e antes de qualquer chamada ao Claude sobre estes sinais.

## Pergunta

Um modelo de linguagem que "entende contexto" filtra os sinais melhor que o gradient boosting de C6?

## Sinais

Os mesmos da prova principal de C6 (grupo B, 31 mercados; entradas de 2025-09-01 a 2026-08-31), restritos às estratégias de barra diária entre as 12: TURTLE20, TURTLE55, SQUEEZE_D1, NR7_D1, CHANDELIER_D1. Os avaliadores de contexto são os de C6, calculados pelo mesmo código.

## Como o Claude decide (fixo)

- Claude Code headless (`claude -p`) pela assinatura do Igor, modelo `opus`, sem ferramentas.
- **Anonimizado:** o Claude não recebe nome do mercado, data nem preço, só a estratégia, o lado, a classe do mercado (câmbio, índice, metal, energia) e os avaliadores numéricos. Sem isso, o modelo poderia "lembrar" o que o mercado fez em 2025-2026, que está dentro do período de treino dele.
- Lotes de 25 sinais por chamada, em ordem aleatória (semente 20260926); resposta em JSON com "entrar" ou "pular" por sinal. Instrução fixa em `backend/scripts/research/fx_c7_llm_banca.py`, commitada antes da primeira chamada.
- Sinal sem resposta válida depois de 3 tentativas conta como "pular".

## Critério

Aprovado se, nos sinais que o Claude aceita: média > 0 no custo base e no estresse; valor-p unilateral do t por cluster de dia ≤ 0,05; ao menos 50 aceitos. Informativo: comparação com a banca do C6 e com todos os sinais, no mesmo conjunto.
