import json
from pathlib import Path

from caudal import points
from caudal.sources import nldi

HOPLAND = Path(__file__).parent / "fixtures" / "nldi_basin_11462500.json"  # the outline NLDI returned on 2026-11-17


def test_network_names_existing_points_and_ends():
    known = set(points.POINTS) | set(points.GAUGES)
    for lid, down in points.DOWNSTREAM.items():
        assert lid in known and down in known
        seen = {lid}
        while down is not None:
            assert down not in seen, f"{lid} drains back to itself"
            seen.add(down)
            down = points.DOWNSTREAM.get(down)


def test_upstream_of_guerneville_is_healdsburg_and_lake_sonoma():
    assert sorted(points.upstream("GUEC1")) == ["HEAC1", "WSDC1"]
    assert points.upstream("HEAC1") == ["HOPC1"] and points.upstream("LAMC1") == []
    assert [points.OUTFLOW[p] for p in sorted(points.upstream("GUEC1"))] == ["11464000", "11465350"]


def test_rain_points_of_hopland_are_the_lattice_inside_its_basin():
    assert points.RAIN_POINTS["HOPC1"] == nldi.lattice(nldi.outline(json.loads(HOPLAND.read_text())))
    assert set(points.RAIN_POINTS) == set(points.GAUGES)
    assert set(points.RAIN_POINTS["HOPC1"]) < set(points.RAIN_POINTS["HEAC1"]) <= set(points.RAIN_POINTS["GUEC1"])
