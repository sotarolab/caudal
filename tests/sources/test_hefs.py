from pathlib import Path

import numpy as np
import pandas as pd

from caudal.sources import hefs

# The first 8 lines of CNRFC's hourly group file of 2019-02-26 12Z, kept to the first members of two gauges.
ROWS = (Path(__file__).parent.parent / "fixtures" / "hefs_hourly_russiannapa_2019-02-26.csv").read_text().splitlines()


def test_parse_reads_a_gauge_s_members_as_quantiles_in_cfs():
    f = hefs.parse(ROWS, "GUEC1")
    members = np.array([[float(v) for v, c in zip(r.split(",")[1:], ROWS[0].split(",")[1:]) if c == "GUEC1"] for r in ROWS[2:]])
    assert list(f.columns) == hefs.COLS and f.index[0] == pd.Timestamp("2019-02-26 12:00") and len(f) == 6
    assert np.allclose(f["hefs_q50"].to_numpy(), np.median(members, axis=1) * 1000)  # kcfs -> cfs
    cut = [ROWS[0], ROWS[1], ROWS[2].rsplit(",", 1)[0] + ",-999"]  # a member with no value is left out
    assert hefs.parse(cut, "HOPC1").shape == (1, 5)


def test_an_issue_is_used_only_once_it_was_public():
    starts = pd.to_datetime(["2019-02-25 12:00", "2019-02-26 12:00"])
    issues = pd.DataFrame({"hefs_q50": [1.0, 2.0]}, index=pd.MultiIndex.from_arrays([starts, starts], names=["issue", "time"]))
    hours = pd.to_datetime(["2019-02-26 18:00", "2019-02-26 19:00", "2019-02-25 18:00"])
    known = hefs.known_at(issues, hours)
    assert list(known) == [starts[0], starts[1], pd.NaT] or (known.iloc[0] == starts[0] and known.iloc[1] == starts[1] and pd.isna(known.iloc[2]))
