"""
panel.py — assemble the per-symbol indicator frames and the cross-sectional snapshot.

This is the bridge between "rows in SQLite" and "a FactorContext the factor
modules can reason about". It also computes the lagged copies of a few
indicators (sma50 twenty days ago, MACD five days ago) that the technical block
needs to detect *changes* — a golden cross is an event, not a state.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .indicators import compute_all

log = logging.getLogger("nsealpha.panel")

LAGGED = {
    "sma50_prev20": ("sma50", 20),
    "sma200_prev20": ("sma200", 20),
    "macd_prev5": ("macd", 5),
    "macd_signal_prev5": ("macd_signal", 5),
    "close_prev5": ("close", 5),
}


def build_frames(prices: pd.DataFrame, symbols: Iterable[str],
                 as_of: Optional[pd.Timestamp] = None,
                 progress=None) -> Dict[str, pd.DataFrame]:
    """symbol -> date-indexed frame with every indicator column."""
    symbols = list(symbols)
    wanted = set(symbols)
    p = prices[prices["symbol"].isin(wanted)].copy()
    p["date"] = pd.to_datetime(p["date"])
    if as_of is not None:
        p = p[p["date"] <= as_of]
    p = p.sort_values(["symbol", "date"])

    frames: Dict[str, pd.DataFrame] = {}
    for i, (sym, g) in enumerate(p.groupby("symbol", sort=False)):
        g = g.drop_duplicates("date").set_index("date")
        if len(g) < 30:
            continue
        try:
            f = compute_all(g)
        except Exception as e:                     # a broken series must not kill the run
            log.debug("indicator failure on %s: %s", sym, e)
            continue
        for new, (src, lag) in LAGGED.items():
            f[new] = f[src].shift(lag) if src in f else np.nan
        frames[sym] = f
        if progress and i % 200 == 0:
            progress(i + 1, len(symbols), sym)
    log.info("built indicator frames for %d symbols", len(frames))
    return frames


def build_snapshot(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Latest row of every frame, stacked into one wide cross-sectional table."""
    rows = {}
    for sym, f in frames.items():
        if f.empty:
            continue
        last = f.iloc[-1]
        rows[sym] = last
    snap = pd.DataFrame.from_dict(rows, orient="index")
    snap.index.name = "symbol"
    if "date" in snap.columns:
        snap = snap.drop(columns=["date"])
    return snap


def sector_map_from_fundamentals(fund: pd.DataFrame) -> Dict[str, str]:
    if fund is None or fund.empty or "symbol" not in fund.columns:
        return {}
    col = "sector" if "sector" in fund.columns else (
        "industry" if "industry" in fund.columns else None)
    if col is None:
        return {}
    m = fund.set_index("symbol")[col].dropna().astype(str)
    return m[m.str.len() > 0].to_dict()


def latest_trading_date(prices: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(prices["date"]).max()
