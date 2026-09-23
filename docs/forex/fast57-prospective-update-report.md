# Fast57 — atualizador prospectivo

Data: 2026-09-23  
Status: automação implementada e rodada; ainda sem novo dia UTC completo após 2026-09-22.  
Escopo: research-only.

## O que foi criado

Script:

`backend/scripts/research/run_fast57_prospective_update.py`

Ele executa a rotina segura do Fast57:

1. calcula o último dia UTC completo como "ontem UTC";
2. baixa `aggTrades` públicos diários completos quando faltarem;
3. materializa partições low-cost com a mesma configuração do candidato congelado;
4. preserva o bloco seed `2026-09-21` a `2026-09-22`;
5. acrescenta blocos prospectivos a partir de `2026-09-23`;
6. chama o runner oficial `run_fast57_fixed_density_prospective_gate.py`;
7. mantém `approved_strategy=false` sempre.

Flags de segurança mantidas:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Comando validado

```bash
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_fast57_prospective_update.py
```

## Resultado em 2026-09-23

Como `2026-09-23` ainda não é um dia UTC completo, o updater não tentou abrir
dados parciais:

- `new_complete_dates=[]`;
- `start_date=2026-09-23`;
- `end_date=2026-09-22`.

Relatório local ignorado pelo git:

`backend/data/microstructure_v1/reports/fast57-fixed-density-prospective-update-current/update-report.json`

Hash:

`39cd83b418c5a3d0ea63e95584f5fd8176301d9c2c0dcceceea3e436ef9572ea`

Relatório Fast57 gerado pelo updater:

`backend/data/microstructure_v1/reports/fast57-fixed-density-prospective-update-current/fast57/report.json`

Hash:

`015ea53d0146b5107ef5a6d9c366ae0c4332dfa95950374e247623d1ee77cf9f`

## Métricas atuais

| Métrica | Gate `5min≤0` | Sem gate |
| --- | ---: | ---: |
| Dias completos | 2 | 2 |
| Trades | 28 | 121 |
| Média 1 bps | +3,4290 | +0,8962 |
| Média 2 bps | +2,4290 | -0,1038 |
| Total 2 bps | +68,0117 | -12,5609 |
| Max DD | 31,2116 | 123,5421 |

Status:

`FAST57_INSUFFICIENT_PROSPECTIVE_SAMPLE`

Motivos:

- `complete_days_below_minimum:2<10`;
- `trade_count_below_minimum:28<100`.

## Decisão

Ainda não há estratégia aprovada.

O updater remove a parte manual perigosa: quando novos arquivos diários completos
existirem, ele consegue baixar, materializar e reavaliar o candidato sem mexer em
modelo, threshold, top-p, gate ou trailing.

Próximo passo operacional:

1. rodar o updater depois que `2026-09-23` fechar e o arquivo público da Binance
   estiver disponível;
2. repetir até atingir pelo menos 10 dias completos e 100 trades;
3. só então interpretar os critérios de confirmação;
4. mesmo se passar, seguir apenas para book/shadow, não Testnet/LIVE.

Sem Testnet/LIVE.
