"""
factors/ — the five analytical blocks.

Every module exposes:

    compute(ctx) -> pd.DataFrame indexed by symbol

where the returned columns are *raw sub-factor values, oriented so that higher
is better*. Shaping (e.g. "RSI 60 is better than RSI 85") happens inside the
module; cross-sectional standardisation happens later in scoring.py. That split
keeps the finance in the factor files and the statistics in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class FactorContext:
    """Everything the factor modules are allowed to look at."""
    frames: Dict[str, pd.DataFrame]      # symbol -> full indicator frame (date-indexed)
    snapshot: pd.DataFrame               # symbol -> latest row of every indicator
    benchmark: pd.DataFrame              # date, close for the benchmark index
    sector_indices: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sector_map: Dict[str, str] = field(default_factory=dict)   # symbol -> sector
    fundamentals: pd.DataFrame = field(default_factory=pd.DataFrame)
    financials: pd.DataFrame = field(default_factory=pd.DataFrame)   # statement metrics
    announcements: pd.DataFrame = field(default_factory=pd.DataFrame)
    deals: pd.DataFrame = field(default_factory=pd.DataFrame)
    fno: pd.DataFrame = field(default_factory=pd.DataFrame)          # per-symbol derivatives
    shareholding: pd.DataFrame = field(default_factory=pd.DataFrame) # QoQ ownership
    market_flows: dict = field(default_factory=dict)                 # FII/DII, participant OI
    config: dict = field(default_factory=dict)
    as_of: Optional[pd.Timestamp] = None

    @property
    def symbols(self):
        return list(self.snapshot.index)
