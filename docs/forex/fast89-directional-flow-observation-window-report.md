# Fast89 — janela prospectiva para Fast88

Resposta objetiva: a janela Fast89 não gerou estratégia nem trade candidato.
Ela capturou mercado real depois do pré-registro Fast88, mas o top-p congelado
não acionou nenhuma entrada.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Captura WAL

Janela: `prospective-wal-fast89-30m`.

Volume capturado em `2026-09-23`:

| Stream | Eventos |
|---|---:|
| depth | `16.912` |
| mark_price | `1.380` |
| spot_trade | `32.658` |
| trade | `30.389` |

## Fast84 — score do top-p congelado

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast84_top_p_silence_diagnostic.py \
  --wal-roots fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m \
  --output-root backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast89
```

Resultado:

- status: `FAST84_TOP_P_NEAR_THRESHOLD_RESEARCH_ONLY`;
- scored rows: `666`;
- top-p entries: `0`;
- threshold: `0.90195311025804`;
- max probability: `0.8613485017346285`;
- distance from threshold at max probability: `0.04060460852341152`;
- rows within `0.05` do threshold: `5`;
- rows within `0.10` do threshold: `229`.

## Fast88 — observação directional-flow

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast88_directional_flow_observation.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast89/all_scored_decisions.jsonl \
  --wal-roots fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m \
  --output-root backend/data/microstructure_v1/reports/fast88-directional-flow-observation-fast89
```

Resultado:

- status: `FAST88_OBSERVATION_NO_TOP_P_ENTRIES_YET`;
- input records: `666`;
- post-preregistration records: `666`;
- candidate universe: `0`;
- rule matches: `0`;
- eligible observations: `0`;
- stress total: `0.0` bps;
- testnet/live: `false`.

## Interpretação

Fast89 não prova que a hipótese Fast88 é ruim; prova uma coisa mais simples:
nesta janela de 30 minutos, o modelo congelado não encontrou entrada com
probabilidade suficiente para sequer testar o trailing.

Então o resultado não é “perdeu dinheiro”. O resultado é “não operou”.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O estado atual continua:

1. Fast80 direto rejeitado;
2. Fast87 encontrou pista pequena e não robusta;
3. Fast88 pré-registrou a pista;
4. Fast89 capturou uma janela prospectiva real;
5. Fast89 não teve entrada top-p;
6. precisamos de mais janelas prospectivas ou de uma nova família de sinal.

## Arquivos e hashes

- Fast84 report:
  `d9d7386464bd05470a51a980fd469ba8839b050bc1c4ad67eefa74acf7dedff3`;
- Fast84 all-scored decisions:
  `ccedd0dadfdd7bbbd74d8060cf7473f15e4a67f1a8342db888e94b1596f99634`;
- Fast88 report:
  `9f976b28aa2b2c7db5fdbee7526159422d9ac7555c6c754582188985e84ae817`;
- Fast88 observation files:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
  para rule matches, conflict-resolved candidates, settlements e observations
  vazios.

## Próximo passo real

Continuar capturando janelas prospectivas e rodar Fast84/Fast88 enquanto a regra
ficar congelada. Se várias janelas seguirem sem top-p, o problema deixa de ser
trailing/stop e passa a ser geração de entradas: a família do modelo atual está
conservadora demais para day trade curto nesse regime.
