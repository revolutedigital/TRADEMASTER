"""C7: nothing that identifies the market or the date reaches the model, and answers are parsed strictly."""

import pandas as pd

from scripts.research import fx_c7_llm_banca as c7


def test_the_rows_sent_to_the_model_carry_no_market_date_or_price() -> None:
    frame = pd.DataFrame([{"config": "TURTLE20", "side": 1.0, "symbol": "SPXUSD", "entry_time": 1.79e9, "net": 2.0,
                           "stress": 1.0, "cost_atr": 0.01, "vol_d1": 1.2, "vol_h1": 0.9, "trend50": 1.0, "trend200": 2.0,
                           "slope50": 0.1, "trend50_side": 1.0, "trend200_side": 2.0, "slope50_side": 0.1, "rsi14": 60.0,
                           "ret4h_side": 0.2, "recent_same": float("nan"), "recent_config": -0.1, "hour": 17, "weekday": 2}])

    (row,) = c7.anonymised(frame)

    text = str(row)
    assert "SPX" not in text and "1790000000" not in text and "net" not in row and "stress" not in row
    assert row["classe"] == "índice de ações" and row["recent_same"] is None


def test_answers_must_cover_every_id_or_are_rejected() -> None:
    assert c7.parse('ok [{"id": 1, "decisao": "entrar"}, {"id": 2, "decisao": "pular"}]', {1, 2}) == {1: True, 2: False}
    assert c7.parse('[{"id": 1, "decisao": "entrar"}]', {1, 2}) is None
    assert c7.parse("não sei", {1}) is None
