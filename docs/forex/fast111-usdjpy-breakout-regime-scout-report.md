# Fast111 — scout de regime para USDJPY London breakout

Resposta objetiva: ainda não temos estratégia aprovada. Mas a pista ficou mais
clara.

Fast110 deixou uma linha weak:

```text
USDJPY
F1a / London breakout
range width 0.6-2.0 ATR
signal_end 11:00
exit 16:30
reward_risk 1.0, 1.5, 2.5
```

Fast111 testou filtros simples, conhecidos antes/na entrada:

- lado da entrada (`LONG`/`SHORT`);
- dia da semana em Londres;
- hora local de entrada em Londres;
- bucket de stop (`narrow`, `middle`, `wide`) definido pelos quantis do S0;
- combinações simples de duas dimensões.

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

## Escopo

Fonte:

- weak survivors do Fast110:
  `backend/data/lab_fast110/fx-fast2-pair-validation/weak_validation_survivors.json`;
- source sha256:
  `efccc47f8f5d453d870540c8bb6aa514eba44044ee3e136a2b3d805196578fb7`.

Escopo:

| Métrica | Valor |
|---|---:|
| Par | `USDJPY` |
| Configurações | `3` |
| Filtros por config | `66` |
| Filtros selecionados no S0 | `7` |
| Weak survivors em S1+S2 | `5` |
| Strict survivors | `0` |
| Estratégia aprovada | `false` |

Status:

`FAST111_USDJPY_BREAKOUT_REGIME_WEAK_ONLY_FAILED_STRICT_STABILITY`

## Melhor filtro encontrado

O melhor padrão foi:

```text
london_entry_hour=8
stop_bucket=wide
```

Interpretação: dentro da hipótese USDJPY London breakout, os rompimentos que
entram logo na hora 08:00 de Londres e têm stop/range mais largo foram melhores
que o conjunto bruto.

| Config | S0 trades | S0 stress R | S0 stress years | S0 t | S1 trades | S1 stress R | S1 stress years | S1 t | S2 trades | S2 stress R | S2 stress years | S2 t |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `F1a.rr2.5.w0.6-2.0.se11:00.x16:30` | `26` | `+0.093614953450405` | `4/5` | `0.8462093653696644` | `26` | `+0.02441183149637864` | `3/5` | `0.1806639730230365` | `36` | `+0.10626468772283898` | `2/3` | `1.1826042848176865` |
| `F1a.rr1.5.w0.6-2.0.se11:00.x16:30` | `26` | `+0.05582413798346849` | `4/5` | `0.6767233727277767` | `26` | `+0.03530915661356638` | `4/5` | `0.253980991042361` | `36` | `+0.10337881880507503` | `2/3` | `1.1666408393477703` |
| `F1a.rr1.0.w0.6-2.0.se11:00.x16:30` | `26` | `+0.037698693679042765` | `4/5` | `0.5242150656456438` | `26` | `+0.01603370954277766` | `3/5` | `0.12114222491723686` | `36` | `+0.09184202968669922` | `2/3` | `1.0885600365511898` |

## Por que ainda não é estratégia

Apesar de melhorar a média:

- a contagem de trades é pequena;
- ainda há anos negativos;
- o `t` continua baixo;
- o filtro foi escolhido no histórico;
- não há observação prospectiva pré-registrada;
- não há evidência suficiente para liberar Testnet/Demo/LIVE.

O resultado bom em S2 é encorajador, mas ele não apaga a instabilidade do S1 nem
o fato de que a regra foi descoberta por busca.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O que temos agora é uma hipótese estreita e testável:

```text
USDJPY London breakout
entrada às 08h Londres
stop/range wide
reward_risk preferencialmente 2.5
```

Ela merece pré-registro prospectivo. Se passar em dados novos, aí sim vira
candidata real.

## Próximo caminho recomendado

Fast112 deve congelar a regra acima e rodar uma observação prospectiva ou uma
validação temporal adicional sem mexer no filtro:

- par: `USDJPY`;
- config: `F1a.rr2.5.w0.6-2.0.se11:00.x16:30`;
- filtro: `london_entry_hour=8 && stop_bucket=wide`;
- threshold de `stop_bucket=wide` congelado pelo S0;
- sem novas combinações;
- `research_only=true`.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast111_usdjpy_breakout_regime_scout.py \
  --progress
```

## Verificações

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast111_usdjpy_breakout_regime_scout.py -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast111_usdjpy_breakout_regime_scout.py \
  backend/tests/unit/test_fast111_usdjpy_breakout_regime_scout.py
```

Resultado:

- `5 passed`;
- `All checks passed!`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast111/usdjpy-breakout-regime-scout/report.json`;
- report sha256:
  `29967566629b6764a42be9289de02bdf835ba8df055fcc138034b6cedefe0f4b`;
- filter evaluations sha256:
  `bfcadb98ebe205af8394f49932f3adbed1d4556dacf6959b70a67173c7585ddc`;
- selected filters sha256:
  `39eec14c3aa8352f2f1c331d5a9afa6e19ea5d192baaa224da544913f7a15d06`;
- weak survivors sha256:
  `0f2da3a9600681c73346dc5abf9dda794f693d2d0f68e5e5f76ff6c62880f57c`;
- strict survivors sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- trade rows sha256:
  `4093661a12ce29ddd84c50b5ab24c7cd8f78bcf35127966edfbf1c11e256380f`.
