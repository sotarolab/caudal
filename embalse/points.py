"""Forecast points, their observation source, location and regulation treatment.

Regulation decides what the model is asked to predict:
- "inflow":  reservoir inflow, upstream of the dam, so the river is treated as natural (CDEC sensor 76 is the truth).
- "natural": a gauge with no major upstream regulation; predict the flow directly.
- "handoff": a gauge below a dam; recent observed flow is fed to the model so it learns the release level (v2).

Coordinates and elevation (ft) are the CDEC station's, i.e. the dam. WRS, DNP, CLE, INV have negative computed
inflows at low flow and are archived in inflow-ledger but not scored or trained on. HETC1 and CMPC1 have no CDEC inflow.
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
