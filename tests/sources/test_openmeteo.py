from datetime import date

import pandas as pd

from caudal.sources import openmeteo


def answer(start: str, hours: int, rain: float = 1.0) -> dict:
    t = pd.date_range(start, periods=hours, freq="h")
    return {"hourly": {"time": [x.strftime("%Y-%m-%dT%H:%M") for x in t], "precipitation": [rain] * hours,
                       "temperature_2m": [10.0] * hours}}


def test_parse_reads_one_point_as_an_hourly_frame():
    a = answer("2019-02-26T00:00", 3)
    a["hourly"]["precipitation"][2] = a["hourly"]["temperature_2m"][2] = None  # an hour the service has not filled
    w = openmeteo.parse(a)
    assert list(w.columns) == ["precip", "temp"] and list(w.index) == list(pd.date_range("2019-02-26", periods=2, freq="h"))
    assert w.precip.tolist() == [1.0, 1.0] and w.temp.tolist() == [10.0, 10.0]


def test_update_archive_extends_and_keeps_what_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(openmeteo, "WEATHER_DIR", tmp_path)
    monkeypatch.setattr(openmeteo, "START", date(2019, 1, 1))
    urls = []

    def get(url):
        urls.append(url)
        start, end = (url.split(f"{k}_date=")[1][:10] for k in ("start", "end"))
        return answer(f"{start}T00:00", (pd.Timestamp(end) - pd.Timestamp(start)).days * 24 + 24)

    first = openmeteo.update_archive(39.25, -123.0, date(2019, 1, 10), get)
    second = openmeteo.update_archive(39.25, -123.0, date(2019, 1, 12), get)
    assert len(first) == 240 and len(second) == 288 and first.index.isin(second.index).all()
    assert "start_date=2019-01-03" in urls[1]  # the newest week is read again
    assert openmeteo.load(39.25, -123.0).equals(second)


def test_basin_mean_is_over_the_hours_every_point_has():
    a = openmeteo.parse(answer("2019-01-01T00:00", 4, rain=1.0))
    b = openmeteo.parse(answer("2019-01-01T02:00", 4, rain=3.0))
    m = openmeteo.basin_mean([a, b])
    assert list(m.index) == list(pd.date_range("2019-01-01 02:00", periods=2, freq="h")) and m.precip.tolist() == [2.0, 2.0]


def test_forecast_gives_up_after_three_quick_tries(monkeypatch):
    asked, waits = [], []

    def urlopen(url, timeout):
        asked.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(openmeteo.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(openmeteo.time, "sleep", waits.append)
    try:
        openmeteo.forecast(39.25, -123.0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("a feed that never answers gave a forecast")
    assert asked == [60, 60, 60] and waits == [5, 10]  # three minutes at most, not the archive's 45
