"""Basin outlines from the USGS network-linked data index (NLDI), and the rainfall sample points of a basin:
the points of a regular latitude-longitude lattice that fall inside its outline.

shortcut: lattice points stand in for the basin mean of the rainfall; replace them with an area-weighted mean
over the grid cells when a gridded rainfall source is added.
"""

from __future__ import annotations

import json
import urllib.request

BASIN = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}/basin?f=json"
STEP = 0.125  # degrees: half a cell of ERA5, 14 km north to south at these latitudes


def outline(geojson: dict) -> list[tuple[float, float]]:
    """The outer ring of the basin as (longitude, latitude) pairs; of the largest part, if it has several."""
    g = geojson["features"][0]["geometry"]
    parts = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    return [(x, y) for x, y in max((p[0] for p in parts), key=len)]


def basin(site: str) -> list[tuple[float, float]]:
    with urllib.request.urlopen(BASIN.format(site=site), timeout=120) as r:
        return outline(json.load(r))


def inside(ring: list[tuple[float, float]], lon: float, lat: float) -> bool:
    """Whether the point is in the ring: a ray to the east crosses its edge an odd number of times."""
    crossings = 0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            crossings += 1
    return crossings % 2 == 1


def lattice(ring: list[tuple[float, float]], step: float = STEP) -> list[tuple[float, float]]:
    """The (latitude, longitude) points on multiples of `step` inside the ring, north to south, west to east."""
    lons, lats = [x for x, _ in ring], [y for _, y in ring]
    cols = range(int(min(lons) // step), int(max(lons) // step) + 2)
    rows = range(int(max(lats) // step) + 1, int(min(lats) // step) - 1, -1)
    return [(j * step, i * step) for j in rows for i in cols if inside(ring, i * step, j * step)]
