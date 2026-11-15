"""Forecast points and their regulation treatment.

Regulation decides what the model is asked to predict:
- "inflow":  reservoir inflow, upstream of the dam, so the river is treated as natural (CDEC sensor 76 is the truth).
- "natural": a gauge with no major upstream regulation; predict the flow directly.
- "handoff": a gauge below a dam; recent observed flow is fed to the model so it learns the release level (v2).

Scored CDEC inflow records: COY, ORO, SHA, FOL, BUL, NML, MIL, EXC, ISB, TRM, SCC. WRS, DNP, CLE, INV have
negative computed inflows at low flow and are archived but not scored; HETC1 and CMPC1 have no CDEC inflow.
"""

INFLOW = {  # CNRFC id -> CDEC station with observed daily inflow
    "LAMC1": "COY", "ORDC1": "ORO", "SHDC1": "SHA", "FOLC1": "FOL", "NBBC1": "BUL", "NMSC1": "NML",
    "FRAC1": "MIL", "EXQC1": "EXC", "ISAC1": "ISB", "TMDC1": "TRM", "SCSC1": "SCC",
    "WSDC1": "WRS", "NDPC1": "DNP", "CEGC1": "CLE", "INVC1": "INV",
}
# shortcut: first-pass regulation labels from basin knowledge, confirm against dam records before v2 training
REGULATION = {lid: "inflow" for lid in INFLOW}
REGULATION |= {lid: "handoff" for lid in ["HOPC1", "HEAC1", "GUEC1"]}  # Russian River below Mendocino / Sonoma
REGULATION |= {lid: "natural" for lid in ["NVRC1", "FTSC1", "SCOC1", "LEGC1", "BRGC1", "ARCC1", "ORIC1",
                                          "FTDC1", "MHBC1"]}
