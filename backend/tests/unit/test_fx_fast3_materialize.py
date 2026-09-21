"""Small invariants of the round-3 materializer."""

import pandas as pd

from scripts.research.fx_fast3_materialize import _compact


def test_compact_keeps_boolean_columns_and_downcasts_float_features() -> None:
    frame = pd.DataFrame({"score": [1.0, 2.0], "eligible": [True, False]})

    compact = _compact(frame)

    assert compact["score"].dtype == "float32"
    assert compact["eligible"].dtype == bool
