from __future__ import annotations

import pandas as pd
import pyarrow as pa

from ibis.backends.chalkdf import Backend


def test_select_projection():
    con = Backend()
    con.do_connect()

    arrow_table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    t = con.create_table("t", arrow_table)

    result = t.select("a").execute()

    expected = pd.DataFrame({"a": [1, 2, 3]})
    pd.testing.assert_frame_equal(result, expected)
