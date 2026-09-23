# Rodada 27 — liquidação dos sinais shadow por bid/ask

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast24  
Resultado: os 52 sinais emitidos pelo shadow dry-run foram liquidados por replay executável; nenhuma ordem foi enviada

## Pergunta

Rodada 26 provou que a política fixa consegue emitir sinais em modo online/dry-run. Faltava fechar o
ciclo:

```text
eventos cronológicos → shadow_decisions.jsonl → settlement por bid/ask → métricas
```

Sem reconsultar a regra. O settler lê os sinais emitidos e liquida exatamente esses sinais.

## O que foi implementado

Script novo:

`backend/scripts/research/settle_book_fixed_gate_shadow_decisions.py`

Ele:

- lê o relatório do shadow dry-run;
- valida `research_only=true`;
- valida `order_submission_allowed=false`;
- valida `execution_authorization=none`;
- exige `committed=false` e `dry_run=true`;
- lê `shadow_decisions.jsonl`;
- liquida só `would_enter=true`;
- usa bid/ask top-of-book;
- reporta `independent` e `one_position`;
- não grava ledger;
- não envia ordem.

## Resultado Fast24 — 100ms de latência

Input:

`backend/data/microstructure_v1/reports/book-fixed-gate-shadow-policy-fast24-dry-run/report.json`

Output:

`backend/data/microstructure_v1/reports/book-fixed-gate-shadow-settlement-fast24-dry-run-100ms/report.json`

Resumo:

| Modo | Sinais shadow | Trades executáveis | Gross médio | Net 0,5 bps | Net 1 bps | Net 2 bps | Saídas |
|---|---:|---:|---:|---:|---:|---:|---|
| Independent | 52 | 52 | +1,340 bps | +0,840 bps | +0,340 bps | -0,660 bps | 20 TARGET / 32 TIMEOUT |
| One-position | 52 | 14 | +1,469 bps | +0,969 bps | +0,469 bps | -0,531 bps | 6 TARGET / 8 TIMEOUT |

Isso bate com o replay executável da Rodada 25, mas agora a fonte são os sinais shadow emitidos pela
política online/dry-run.

## Decisão

A candidata está pronta para um shadow prospectivo contínuo.

Ainda não é estratégia aprovada para Testnet real nem LIVE.

O que agora está fechado:

1. seleção congelada;
2. emissão online/dry-run;
3. liquidação dos sinais emitidos;
4. replay bid/ask com restrição one-position;
5. boundary de segurança sem execução.

O que ainda bloqueia:

- a amostra one-position tem só 14 trades;
- em 2 bps o resultado ainda é negativo;
- isso ainda é replay em WAL, não fluxo vivo de vários dias;
- ainda não há partição prospectiva longa de shadow;
- não há aprovação de Testnet/LIVE.

Próximo passo correto:

1. rodar shadow prospectivo por várias janelas/dias;
2. liquidar os sinais pelo settler;
3. exigir consistência líquida com custo realista;
4. só então discutir Testnet.

Nenhuma ordem foi enviada.

## Artefatos

- Settler:
  `backend/scripts/research/settle_book_fixed_gate_shadow_decisions.py`
- Testes:
  `backend/tests/unit/test_book_fixed_gate_shadow_settlement.py`
- Settlement report:
  `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-settlement-fast24-dry-run-100ms/report.json`
  (`8fa9abeeb7730ea510385452eb748f1801ccd91bc35e11e2b154b977f176ad4e`)
- Settlements independent:
  `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-settlement-fast24-dry-run-100ms/shadow_settlements.parquet`
  (`b63e4ac6bd9a29800c493c2293ddeee8490573b28d6faabbbc5608a485df9229`)
- Settlements one-position:
  `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-settlement-fast24-dry-run-100ms/one_position_settlements.parquet`
  (`fd70e036273bbfb7688813c96a7c37f6e6d836b8d46e44f14743cd428e27ceaf`)
