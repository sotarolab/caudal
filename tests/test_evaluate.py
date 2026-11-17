import pandas as pd

from caudal.evaluate import persistence


def test_persistence_is_the_newest_inflow_the_forecast_could_see():
    days = pd.date_range("2020-01-01", periods=10, freq="D")
    inflow = pd.Series(range(10), index=days, dtype=float)
    assert persistence(inflow, days[[5, 9]], lead=0).tolist() == [4.0, 8.0]  # the day before the target
    assert persistence(inflow, days[[5, 9]], lead=3).tolist() == [1.0, 5.0]  # issued 3 days ahead: 4 days before
