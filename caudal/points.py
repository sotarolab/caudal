"""Forecast points, their observation source, location and regulation treatment.

Regulation decides what the model is asked to predict:
- "inflow":  reservoir inflow, upstream of the dam, so the river is treated as natural (CDEC sensor 76 is the truth).
- "natural": a gauge with no major upstream regulation; predict the flow directly.
- "handoff": a gauge below a dam; recent observed flow is fed to the model so it learns the release level (v2).

Coordinates and elevation (ft) are the CDEC station's, i.e. the dam. WRS, DNP, CLE, INV have negative computed
inflows at low flow and are archived in hapi but not scored or trained on. HETC1 and CMPC1 have no CDEC inflow.

The flood gauges are rows of GAUGES. DOWNSTREAM links every point to the next forecast point down the river,
and RAIN_POINTS lists where each gauge's basin is sampled for rainfall.
"""

# CNRFC id: (CDEC station, name, latitude, longitude, dam elevation ft)
POINTS = {
    "LAMC1": ("COY", "Mendocino", 39.197, -123.186, 720),
    "ORDC1": ("ORO", "Oroville", 39.540, -121.493, 900),
    "SHDC1": ("SHA", "Shasta", 40.718, -122.420, 1067),
    "FOLC1": ("FOL", "Folsom", 38.683, -121.183, 466),
    "NBBC1": ("BUL", "New Bullards Bar", 39.393, -121.140, 1965),
    "NMSC1": ("NML", "New Melones", 37.948, -120.525, 1135),
    "FRAC1": ("MIL", "Millerton", 37.001, -119.705, 581),
    "EXQC1": ("EXC", "McClure", 37.585, -120.270, 879),
    "ISAC1": ("ISB", "Isabella", 35.646, -118.473, 2635),
    "TMDC1": ("TRM", "Kaweah", 36.415, -119.002, 752),
    "SCSC1": ("SCC", "Success", 36.057, -118.924, 692),
    "WSDC1": ("WRS", "Sonoma", 38.723, -123.010, 440),
    "NDPC1": ("DNP", "Don Pedro", 37.702, -120.421, 830),
    "CEGC1": ("CLE", "Trinity", 40.801, -122.762, 2370),
    "INVC1": ("INV", "Indian Valley", 39.083, -122.533, 1502),
}
NOISY = {"WSDC1", "NDPC1", "CEGC1", "INVC1"}
TRAINED = [lid for lid in POINTS if lid not in NOISY]

# shortcut: first-pass regulation labels from basin knowledge, confirm against dam records before v2 training
REGULATION = {lid: "inflow" for lid in POINTS}
REGULATION |= {lid: "handoff" for lid in ["HOPC1", "HEAC1", "GUEC1"]}  # Russian River below Mendocino / Sonoma
REGULATION |= {lid: "natural" for lid in ["NVRC1", "FTSC1", "SCOC1", "LEGC1", "BRGC1", "ARCC1", "ORIC1",
                                          "FTDC1", "MHBC1"]}

# CNRFC id: (USGS site that measures its discharge, name, latitude, longitude)
# GUEC1: the NWS gauge is at Johnsons Beach (USGS 11467002), which records stage only. Its discharge is taken
# at Hacienda Bridge, upstream, which drains 1,338 of Johnsons Beach's 1,353 square miles.
GAUGES = {
    "HOPC1": ("11462500", "Russian River near Hopland", 39.027, -123.131),
    "HEAC1": ("11464000", "Russian River near Healdsburg", 38.613, -122.836),
    "GUEC1": ("11467000", "Russian River at Hacienda Bridge near Guerneville", 38.508, -122.928),
}

# The river network: each point names the next forecast point downstream. Dry Creek, with Lake Sonoma on it,
# joins the Russian River below the Healdsburg gauge (USGS drainage areas in square miles: Healdsburg 793,
# Dry Creek near mouth 217, Hacienda Bridge 1,338).
DOWNSTREAM = {"LAMC1": "HOPC1", "HOPC1": "HEAC1", "HEAC1": "GUEC1", "WSDC1": "GUEC1"}

# The USGS site that measures what a point sends downstream: a gauge's own site, and Dry Creek near its mouth
# for Lake Sonoma. Lake Mendocino has none here: no gauge below its dam was checked.
OUTFLOW = {**{lid: row[0] for lid, row in GAUGES.items()}, "WSDC1": "11465350"}


def upstream(lid: str) -> list[str]:
    """The points that drain straight to this one."""
    return [p for p, d in DOWNSTREAM.items() if d == lid]


# (latitude, longitude) of the rainfall sample points of each gauge's whole basin: the 0.125 degree lattice
# points inside the outline NLDI returns for the gauge's USGS site. Written by scripts/basin_points.py on
# 2026-11-17. A basin holds the points of the basins upstream of it.
RAIN_POINTS = {
    "HOPC1": [  # 7 points, basin of USGS 11462500
        (39.375, -123.25), (39.375, -123.125), (39.25, -123.25), (39.25, -123.125), (39.25, -123.0),
        (39.125, -123.25), (39.125, -123.125),
    ],
    "HEAC1": [  # 15 points, basin of USGS 11464000
        (39.375, -123.25), (39.375, -123.125), (39.25, -123.25), (39.25, -123.125), (39.25, -123.0),
        (39.125, -123.25), (39.125, -123.125), (39.0, -123.25), (39.0, -123.125), (39.0, -123.0), (38.875,
        -123.0), (38.75, -123.0), (38.75, -122.875), (38.75, -122.75), (38.625, -122.75),
    ],
    "GUEC1": [  # 25 points, basin of USGS 11467000
        (39.375, -123.25), (39.375, -123.125), (39.25, -123.25), (39.25, -123.125), (39.25, -123.0),
        (39.125, -123.25), (39.125, -123.125), (39.0, -123.25), (39.0, -123.125), (39.0, -123.0), (38.875,
        -123.125), (38.875, -123.0), (38.75, -123.125), (38.75, -123.0), (38.75, -122.875), (38.75,
        -122.75), (38.625, -123.0), (38.625, -122.875), (38.625, -122.75), (38.5, -122.875), (38.5,
        -122.75), (38.5, -122.625), (38.375, -122.875), (38.375, -122.75), (38.375, -122.625),
    ],
}
