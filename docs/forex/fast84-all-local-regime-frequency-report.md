# Fast84 — frequência top-p/regime em todos os WALs locais

Resposta objetiva: o setup Fast80 existe nos WALs locais antigos, mas não
apareceu nas duas janelas prospectivas novas Fast82/Fast83.

Este relatório é diagnóstico. Ele mistura janelas anteriores ao pré-registro
com janelas posteriores apenas para medir frequência de regime e entender se a
hipótese é operacionalmente rara. Ele não aprova estratégia e não substitui o
gate prospectivo.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Comando

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast84_top_p_silence_diagnostic.py \
  --output-root backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-all-local-wals
```

## Resultado agregado

- scored rows: `8.026`;
- threshold: `0.90195311025804`;
- top-p entries: `550`;
- would-enter rate: `0.06852728631946176`;
- Fast80 rule matches: `18`;
- Fast80 rule matches que também eram top-p: `15`;
- maior probabilidade: `0.9697745619026708`;
- maior probabilidade entre rule matches Fast80:
  `0.9659557303776568`.

Distribuição agregada:

| Métrica | Probabilidade |
|---|---:|
| min | `0.4655914798346267` |
| p25 | `0.8125253834541083` |
| p50 | `0.8400632109476411` |
| p75 | `0.8683988062942183` |
| p90 | `0.8948847732641846` |
| p95 | `0.9107768973504388` |
| p99 | `0.9366061612663514` |
| max | `0.9697745619026708` |
| mean | `0.8388830724667763` |

## Por janela

| WAL | Scored rows | Top-p | Fast80 matches | Fast80+top-p | Max prob |
|---|---:|---:|---:|---:|---:|
| fast16 | 408 | 0 | 0 | 0 | `0.8613330718546762` |
| fast20 | 696 | 16 | 4 | 2 | `0.9282682480982931` |
| fast21 | 696 | 38 | 2 | 1 | `0.9359062667054268` |
| fast22 | 696 | 2 | 0 | 0 | `0.9055979662880687` |
| fast24 | 696 | 22 | 0 | 0 | `0.9313695809970836` |
| fast28 | 696 | 13 | 0 | 0 | `0.9247019233457621` |
| fast29 | 692 | 51 | 0 | 0 | `0.9347457220050902` |
| fast32 | 690 | 0 | 0 | 0 | `0.8798938984019395` |
| fast35 | 696 | 71 | 0 | 0 | `0.9229169964540247` |
| fast36 | 696 | 337 | 12 | 12 | `0.9697745619026708` |
| fast82 | 678 | 0 | 0 | 0 | `0.89571932347643` |
| fast83 | 686 | 0 | 0 | 0 | `0.8917379594024417` |

## Exemplos de matches Fast80

Fast80 apareceu em janelas de volatilidade alta:

- fast20 `2026-09-23T03:40:55Z`, SELL, probability
  `0.9207065229051108`, vol 30s `58.6319687690583`, flow
  `-0.3873232448266145`;
- fast21 `2026-09-23T04:23:30Z`, SELL, probability
  `0.9119032076692549`, vol 30s `117.62939196462018`, flow
  `-0.4435902187962738`;
- fast36 `2026-09-23T14:13:15Z`, BUY, probability
  `0.9659557303776568`, vol 30s `95.23483026169026`, flow
  `-0.529044292679266`.

Nas janelas novas Fast82/Fast83, as top linhas ficaram perto do threshold, mas
a volatilidade ficou baixa demais para Fast80:

- Fast82 max probability: `0.89571932347643`, vol 30s
  `0.7674571944414029`;
- Fast83 max probability: `0.8917379594024417`, vol 30s
  `6.145093624355911`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Leitura atual:

1. O modelo top-p não está morto: em todo o WAL local, ele gerou 550 entradas.
2. O regime Fast80 existe, mas é raro e concentrado em poucas janelas.
3. As duas janelas prospectivas novas ainda não pegaram esse regime.
4. O próximo critério não deve ser “baixar threshold no impulso”; deve ser:
   coletar mais janelas pós-pré-registro e, quando o regime aparecer, liquidar
   com Fast61/Fast80 sem recalibrar.

## Arquivos e hashes

- Fast84 all-local report:
  `dfd8d6f9e1a9ba4693b8144c678ceae53aeb2fb1efc615f32521d051ff086a4a`;
- all scored decisions:
  `5275975a26478a159a643ec4785a0338732df11a2b7bd3f576e323d05b2a4976`;
- top scored decisions:
  `54601e995fe4f4d164871a0452a1ca0e178e41d309b03403011a459e58c06421`.
