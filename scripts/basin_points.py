"""Print the rainfall sample points of each gauge's basin, as the rows of RAIN_POINTS in caudal/points.py.

    python scripts/basin_points.py        # one request per gauge to the USGS network service
"""

from __future__ import annotations

from caudal.points import GAUGES
from caudal.sources import nldi

if __name__ == "__main__":
    for lid, (site, *_) in GAUGES.items():
        points = nldi.lattice(nldi.basin(site))
        print(f'    "{lid}": {points},  # {len(points)} points, basin of USGS {site}')
