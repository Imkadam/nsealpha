"""
universe.py — decide which stocks are even eligible to be ranked.

This is the least glamorous and most valuable file in the project. On NSE the
main way a screener embarrasses itself is by surfacing a ₹14 stock with ₹40 lakh
of daily turnover that just went up 400% on 9% delivery. Those names look
spectacular on every momentum factor and are untradeable in practice.

Hard gates applied here (a stock fails ANY -> excluded, no scoring):
  series in EQ            BE/BZ/IL = trade-to-trade or surveillance segment
  price >= min_price      no penny stocks
  turnover >= threshold   20-day average traded value, in Rs crore
  history >= N days       factors need history to be meaningful
  traded recently         not suspended / illiquid
  delivery % floor        persistent sub-20% delivery = intraday churn
  not on ASM/GSM          NSE surveillance frameworks
  not an SME/ETF/SGB/REIT
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nsealpha.universe")

# Instruments that show up in the bhavcopy but are not equities we want to rank.
NON_EQUITY_PATTERNS = [
    r"BEES$", r"^GOLD", r"^SILVER", r"ETF$", r"^LIQUID", r"^NIFTY", r"^BANKNIFTY",
    r"IETF$", r"^SDL", r"^GS\d", r"^SGB", r"INVIT$", r"^INVIT", r"REIT$", r"-RE$",
    r"-RR$", r"-PP$", r"^\d",
]
NON_EQUITY_RE = re.compile("|".join(NON_EQUITY_PATTERNS), flags=re.I)

# Series codes that are NOT normal rolling-settlement equity
RESTRICTED_SERIES = {"BE", "BZ", "IL", "XT", "XC", "XD", "SM", "ST", "MT", "M ", "MF",
                     "GB", "GS", "W3", "Y1", "RL", "AF", "AE"}


class UniverseResult:
    def __init__(self, symbols: List[str], rejected: pd.DataFrame, snapshot: pd.DataFrame):
        self.symbols = symbols
        self.rejected = rejected      # symbol + reason, for transparency on the site
        self.snapshot = snapshot      # latest-bar metrics per surviving symbol

    def __len__(self):
        return len(self.symbols)


def _latest_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol summary computed straight off the raw price table (cheap)."""
    prices = prices.sort_values(["symbol", "date"])
    g = prices.groupby("symbol", sort=False)

    last = g.tail(1).set_index("symbol")
    n_days = g.size().rename("n_days")
    # 20-day average turnover in Rs crore
    turn20 = (g["turnover"].apply(lambda s: s.tail(20).mean()) / 1e7).rename("turnover_20d_cr")
    deliv60 = g["deliv_pct"].apply(lambda s: s.tail(60).mean()).rename("deliv_60d")
    # how many of the last 20 sessions actually traded
    active20 = g["volume"].apply(lambda s: (s.tail(20) > 0).sum()).rename("active_20")

    out = pd.concat([last[["date", "series", "close", "volume", "turnover", "deliv_pct"]],
                     n_days, turn20, deliv60, active20], axis=1)
    out.index.name = "symbol"
    return out.reset_index()


def build_universe(prices: pd.DataFrame,
                   cfg: dict,
                   index_members: Optional[List[str]] = None,
                   securities: Optional[pd.DataFrame] = None) -> UniverseResult:
    f = cfg["filters"]
    ucfg = cfg["universe"]
    snap = _latest_metrics(prices)
    last_date = pd.to_datetime(prices["date"]).max()

    reasons: Dict[str, str] = {}

    def reject(mask: pd.Series, why: str):
        for s in snap.loc[mask, "symbol"]:
            reasons.setdefault(s, why)

    keep = pd.Series(True, index=snap.index)

    # --- instrument type ---------------------------------------------------
    bad_name = snap["symbol"].astype(str).str.upper().str.match(NON_EQUITY_RE)
    reject(bad_name, "not a common equity (ETF/SGB/REIT/govt security)")
    keep &= ~bad_name

    allowed_series = {s.upper() for s in ucfg.get("series_allowed", ["EQ"])}
    ser = snap["series"].astype(str).str.strip().str.upper()
    bad_series = ~ser.isin(allowed_series)
    reject(bad_series & keep, "series not in " + "/".join(sorted(allowed_series)) +
           " (trade-to-trade or surveillance segment)")
    keep &= ~bad_series

    # --- index membership --------------------------------------------------
    if index_members:
        members = {m.upper() for m in index_members}
        not_member = ~snap["symbol"].str.upper().isin(members)
        reject(not_member & keep, f"not in {ucfg['source']}")
        keep &= ~not_member

    # --- price -------------------------------------------------------------
    too_cheap = snap["close"] < f["min_price"]
    reject(too_cheap & keep, f"price below Rs {f['min_price']:.0f}")
    keep &= ~too_cheap

    too_dear = snap["close"] > f.get("max_price", 1e9)
    reject(too_dear & keep, "price above configured maximum")
    keep &= ~too_dear

    # --- liquidity ---------------------------------------------------------
    illiquid = snap["turnover_20d_cr"].fillna(0) < f["min_avg_turnover_cr"]
    reject(illiquid & keep,
           f"20d avg turnover below Rs {f['min_avg_turnover_cr']:.1f} cr")
    keep &= ~illiquid

    thin = snap["active_20"].fillna(0) < 18
    reject(thin & keep, "did not trade on at least 18 of the last 20 sessions")
    keep &= ~thin

    # --- history -----------------------------------------------------------
    short = snap["n_days"] < f["min_history_days"]
    reject(short & keep, f"fewer than {f['min_history_days']} days of history")
    keep &= ~short

    stale = (last_date - pd.to_datetime(snap["date"])).dt.days > f.get("max_gap_days", 10)
    reject(stale & keep, "has not traded recently")
    keep &= ~stale

    # --- delivery quality --------------------------------------------------
    if f.get("min_avg_delivery_pct"):
        d = snap["deliv_60d"]
        churn = d.notna() & (d < f["min_avg_delivery_pct"])
        reject(churn & keep,
               f"60d avg delivery below {f['min_avg_delivery_pct']:.0f}% (intraday churn)")
        keep &= ~churn

    # --- surveillance ------------------------------------------------------
    if f.get("exclude_surveillance") and securities is not None and not securities.empty:
        flagged = set(securities.loc[
            securities["surveillance"].fillna("").astype(str).str.len() > 0, "symbol"])
        on_asm = snap["symbol"].isin(flagged)
        reject(on_asm & keep, "on NSE surveillance (ASM/GSM/trade-to-trade)")
        keep &= ~on_asm

    survivors = snap.loc[keep].copy()
    rejected = pd.DataFrame(
        {"symbol": list(reasons.keys()), "reason": list(reasons.values())})
    rejected = rejected[~rejected["symbol"].isin(survivors["symbol"])]

    log.info("universe: %d symbols in, %d survive filters", len(snap), len(survivors))
    return UniverseResult(sorted(survivors["symbol"].tolist()), rejected, survivors)


def rejection_summary(rejected: pd.DataFrame) -> pd.DataFrame:
    if rejected.empty:
        return rejected
    return (rejected.groupby("reason").size()
            .sort_values(ascending=False)
            .rename("count").reset_index())
