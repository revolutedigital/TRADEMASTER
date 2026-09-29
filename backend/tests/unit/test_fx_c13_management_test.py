"""C13-A management side test: the frozen row (g=0) must reproduce the discovery's own net R exactly."""

import numpy as np
import pandas as pd

from scripts.research import fx_c13_banca as banca
from scripts.research import fx_c13_management_test as t


def test_the_frozen_row_reproduces_the_discovery_and_the_tighter_target_touches_more_and_smaller_wins() -> None:
    frame = pd.read_parquet(banca.DATASET, columns=["symbol", "set_id", "entry_time", "fx_day", "net"])
    predictions = pd.read_parquet(banca.PREDICTIONS, columns=["row", "year", t.CONFIG])
    table = frame.iloc[predictions["row"].to_numpy()].reset_index(drop=True)
    table = table.assign(pred=predictions[t.CONFIG].to_numpy(), year=predictions["year"].to_numpy())
    table = table[(table["pred"] > min(t.THRESHOLDS)) & (table["entry_time"] >= t.LAST_DAY - t.WINDOW_DAYS * 86400)]
    table = table.reset_index(drop=True).assign(cid=np.arange(len(table)))
    one = table[table["symbol"] == "EURUSD"]

    result = t.market_worker(("EURUSD", one[["symbol", "set_id", "entry_time", "fx_day", "pred", "year", "cid"]]))
    trades = pd.DataFrame(result, columns=["tau_i", "g", "cid", "net", "stress"])

    frozen = trades[(trades["tau_i"] == t.THRESHOLDS.index(0.05)) & (trades["g"] == 0)]
    merged = frozen.merge(one[["cid", "net"]], on="cid", suffixes=("_new", "_old"))
    assert len(merged) == len(frozen) == len(one[one["pred"] > 0.05])
    assert np.allclose(merged["net_new"], merged["net_old"], atol=1e-9)  # g=0 IS the frozen "alvo 3R"

    tight = trades[(trades["tau_i"] == t.THRESHOLDS.index(0.05)) & (trades["g"] == 1)]
    assert len(tight) >= len(frozen)  # 0.5R fills where 3R would still be running: never fewer exits
    assert (tight["net"] > 0).mean() > (frozen["net"] > 0).mean()  # a closer target is hit more often
    assert tight.loc[tight["net"] > 0, "net"].mean() < frozen.loc[frozen["net"] > 0, "net"].mean()  # but for less
