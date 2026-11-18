import urllib.error
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from caudal.sources import usgs

HOPLAND = Path(__file__).parent.parent / "fixtures" / "usgs_iv_11462500_2019-02-26.rdb"  # one real day, 96 rows
HEAD = "agency_cd\tsite_no\tdatetime\ttz_cd\t16315_00060\t16315_00060_cd\n5s\t15s\t20d\t6s\t14n\t10s\n"


def rdb(*rows: tuple[str, str, str]) -> str:
    return "# a comment line\n" + HEAD + "".join(f"USGS\t11462500\t{t}\t{tz}\t{v}\tA\n" for t, tz, v in rows)


def test_parse_reads_local_times_as_utc():
    q = usgs.parse(rdb(("2019-02-26 00:00", "PST", "11800"), ("2019-07-01 00:15", "PDT", "150")))
    assert list(q.index) == [pd.Timestamp("2019-02-26 08:00"), pd.Timestamp("2019-07-01 07:15")]
    assert list(q) == [11800.0, 150.0]
    with pytest.raises(ValueError, match="time zone"):
        usgs.parse(rdb(("2019-02-26 00:00", "MST", "1")))


def test_parse_reads_a_flagged_value_as_missing():
    q = usgs.parse(rdb(("2019-01-01 00:00", "PST", "Ice"), ("2019-01-01 00:15", "PST", "90.5")))
    assert list(q) == [90.5]
    assert usgs.parse("# no data for this site\n").empty
    assert usgs.parse("agency_cd\tsite_no\tdatetime\n5s\t15s\t20d\n\t\t\n").empty  # what a span with no values gets


def test_hourly_is_the_mean_of_the_hour_ending_at_the_stamp():
    h = usgs.hourly(usgs.parse(HOPLAND.read_text()))
    # 00:15, 00:30, 00:45 and 01:00 PST read 11900, 12100, 12300 and 12400 cfs: the hour ending 01:00 PST, 09:00 UTC
    assert h[pd.Timestamp("2019-02-26 09:00")] == (11900 + 12100 + 12300 + 12400) / 4
    assert h.index[0] == pd.Timestamp("2019-02-26 08:00") and h.iloc[0] == 11800  # 00:00 PST closes the hour before


def test_hourly_leaves_out_an_hour_with_no_value():
    q = usgs.parse(rdb(("2019-01-01 00:15", "PST", "10"), ("2019-01-01 03:00", "PST", "30")))
    assert list(usgs.hourly(q).index) == [pd.Timestamp("2019-01-01 09:00"), pd.Timestamp("2019-01-01 11:00")]


def quarter_hours(start: str, end: str) -> pd.Series:
    """A value every 15 minutes in UTC, equal to the hour of the day it falls in."""
    t = pd.date_range(start, end, freq="15min", name="time")
    return pd.Series(t.hour.astype(float), index=t, name="cfs")


def test_complete_hours_drops_the_partial_first_hour_and_the_hour_in_progress():
    h = usgs.complete_hours("x", date(2019, 1, 2), date(2019, 1, 2), lambda *a: quarter_hours("2019-01-01 08:15", "2019-01-02 10:30"))
    assert h.index[0] == pd.Timestamp("2019-01-01 10:00")  # 09:00 is the first stamp and is dropped
    assert h.index[-1] == pd.Timestamp("2019-01-02 10:00")  # 11:00 has two of its four values


def test_update_archive_extends_and_keeps_what_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(usgs, "FLOW_DIR", tmp_path)
    monkeypatch.setattr(usgs, "START", date(2019, 1, 1))
    asked = []

    def fetch(site, start, end):
        asked.append((start, end))
        return quarter_hours(f"{start} 08:00", f"{end} 23:45")

    first = usgs.update_archive("x", date(2019, 1, 3), fetch)
    assert asked == [(date(2018, 12, 31), date(2019, 1, 3))]  # one request, read from a day before the start
    second = usgs.update_archive("x", date(2019, 1, 5), fetch)
    assert asked[1] == (date(2018, 12, 31), date(2019, 1, 5))  # the last two days are read again, from a day before
    assert second.index[-1] == pd.Timestamp("2019-01-05 23:00") and first.index.isin(second.index).all()
    third = usgs.update_archive("x", date(2019, 1, 5), lambda *a: quarter_hours("2030-01-01", "2030-01-01")[:0])
    assert third.equals(second) and usgs.load("x").equals(second)  # an empty answer is a gap: nothing is lost


def test_days_the_service_refuses_are_left_out_and_the_rest_is_read():
    bad = date(2019, 1, 3)

    refused = urllib.error.HTTPError("https://example", 503, "Service Unavailable", None, None)  # type: ignore[arg-type]

    def fetch(site, start, end):
        if start <= bad <= end:
            raise refused
        return quarter_hours(f"{start} 00:00", f"{end} 23:45")

    q = usgs.fetch_around_bad_days("x", date(2019, 1, 1), date(2019, 1, 8), fetch)
    days = sorted(set(q.index.date))
    assert days == [date(2019, 1, d) for d in (1, 2, 4, 5, 6, 7, 8)]

    def down(site, start, end):
        raise refused

    with pytest.raises(OSError, match="refused every day"):  # nothing read at all: the service is down
        usgs.fetch_around_bad_days("x", date(2019, 1, 1), date(2019, 1, 4), down)
    asked = []

    def silent(site, start, end):
        asked.append((start, end))
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError):  # no answer at all is not a bad day: one request, no halving
        usgs.fetch_around_bad_days("x", date(2019, 1, 1), date(2019, 1, 8), silent)
    assert len(asked) == 1


def test_fetch_passes_its_timeout_to_the_request(monkeypatch):
    seen = {}

    class Answer:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return HOPLAND.read_bytes()

    def urlopen(request, timeout):
        seen["timeout"] = timeout
        return Answer()

    monkeypatch.setattr(usgs.urllib.request, "urlopen", urlopen)
    assert len(usgs.fetch("11462500", date(2019, 2, 26), date(2019, 2, 26), timeout=60)) == 96 and seen["timeout"] == 60
