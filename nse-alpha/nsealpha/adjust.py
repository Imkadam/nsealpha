"""
adjust.py — corporate-action adjustment. Fixes the single most dangerous silent
bug in any Indian equity screener built on bhavcopies.

THE PROBLEM
-----------
NSE's bhavcopy reports actual traded prices. It does not adjust history for
splits, bonuses or rights. So when a stock does a 1:5 split, the raw series shows

    Friday close   2,480
    Monday close     498

and every return-based factor reads that as a -80% day. That poisons:
  * every momentum horizon (1M/3M/6M/12M returns)
  * distance from the 52-week high, and the 52-week high itself
  * ATR, realised volatility, and therefore every stop-loss we suggest
  * the Hurst exponent and the historical-edge factor, which learn from returns
  * OBV, because the sign of the day flips

and it does all of it silently, on exactly the stocks that just rewarded their
shareholders. Left unfixed, the screen systematically hates companies that split.

THE FIX
-------
Two independent sources, cross-checked against each other.

1. PREV_CLOSE (primary). The bhavcopy carries a PREV_CLOSE column, and on an
   ex-date NSE populates it with the *adjusted* previous close — that is what the
   exchange itself uses to compute the day's circuit limits. So for every session:

       implied_factor(t) = prev_close(t) / close(t-1)

   On an ordinary day that ratio is 1.0. On a 1:5 split ex-date it is ~0.2. The
   adjustment factor falls straight out of data we already download, with no
   extra request and no dependence on parsing free text.

2. Corporate actions feed (corroboration). `/api/corporates-corporateActions`
   gives the ex-date and a purpose string like "BONUS 1:1" or "FACE VALUE SPLIT
   FROM RS 10 TO RS 2". We parse the ratio out of that and compare it with the
   implied factor. Agreement raises confidence; disagreement gets logged loudly
   rather than silently resolved, because a mismatch usually means one of the two
   feeds is wrong and you want to know which stock is affected.

Where they disagree beyond tolerance, the parsed corporate action wins — a stated
1:1 bonus is a fact, whereas an implied factor can be corrupted by a bad tick.
Where there is no corporate action on file, the implied factor is used only if it
looks like a real event (a clean, large ratio) rather than rounding noise.

The cumulative factor is then applied backwards through history so that TODAY's
price is always the real, tradable price — you never want the screen quoting an
entry price that no longer exists on the exchange.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nsealpha.adjust")

# A day's implied factor must differ from 1 by more than this to be treated as a
# corporate action rather than a rounding artefact or a stale prev_close.
MIN_EVENT_DEVIATION = 0.02
# How closely the implied factor and the parsed ratio must agree.
CROSSCHECK_TOLERANCE = 0.03
# Sanity bounds — nothing legitimate multiplies a share price by 20x overnight.
FACTOR_BOUNDS = (0.01, 3.0)

PRICE_COLS = ("open", "high", "low", "close", "prev_close", "last_price")


@dataclass
class AdjustmentEvent:
    symbol: str
    date: pd.Timestamp
    factor: float               # multiply pre-event prices by this
    source: str                 # "corporate_action", "prev_close", "both"
    purpose: str = ""
    implied: Optional[float] = None
    parsed: Optional[float] = None

    @property
    def disagrees(self) -> bool:
        if self.implied is None or self.parsed is None:
            return False
        return abs(self.implied - self.parsed) > CROSSCHECK_TOLERANCE


# --------------------------------------------------------------- ratio parsing

_NUM = r"(\d+(?:\.\d+)?)"

RATIO_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # "FACE VALUE SPLIT FROM RS 10/- TO RS 2/-"  -> factor = 2/10 = 0.2
    (re.compile(rf"SPLIT.*?FROM\s*(?:RS\.?|INR)?\s*{_NUM}.*?TO\s*(?:RS\.?|INR)?\s*{_NUM}",
                re.I | re.S), "split_from_to"),
    # "SUB-DIVISION FROM RS 10 TO RE 1"
    (re.compile(rf"SUB[\s-]*DIVISION.*?FROM\s*(?:RS\.?|RE\.?|INR)?\s*{_NUM}.*?TO\s*(?:RS\.?|RE\.?|INR)?\s*{_NUM}",
                re.I | re.S), "split_from_to"),
    # "BONUS 1:1"  -> 1 new share for every 1 held -> factor = 1/(1+1) = 0.5
    (re.compile(rf"BONUS\s*(?:ISSUE)?\s*{_NUM}\s*:\s*{_NUM}", re.I), "bonus"),
    # "BONUS IN THE RATIO OF 2:1"
    (re.compile(rf"BONUS.*?RATIO\s*(?:OF)?\s*{_NUM}\s*:\s*{_NUM}", re.I | re.S), "bonus"),
    # "RIGHTS 1:4 @ PREMIUM" — needs the issue price to be exact; approximated
    (re.compile(rf"RIGHTS?\s*(?:ISSUE)?\s*{_NUM}\s*:\s*{_NUM}", re.I), "rights"),
]


def parse_ratio(purpose: str) -> Tuple[Optional[float], str]:
    """
    Turn NSE's free-text corporate action purpose into a price factor.

    Returns (factor, kind). `factor` multiplies PRE-event prices to put them on
    the post-event scale. None when the text is a dividend or anything else that
    doesn't change the share count.
    """
    if not purpose:
        return None, ""
    text = str(purpose).upper()

    for pat, kind in RATIO_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        a, b = float(m.group(1)), float(m.group(2))
        if kind == "split_from_to":
            # from face value `a` to face value `b`
            if a <= 0:
                return None, kind
            return b / a, kind
        if kind == "bonus":
            # a new shares for every b held -> holder ends with (a+b) per b
            if (a + b) <= 0:
                return None, kind
            return b / (a + b), kind
        if kind == "rights":
            # Without the issue price this is only approximate, and rights are
            # usually priced near market anyway. Skip rather than guess wrong.
            return None, kind
    return None, ""


def is_dividend_only(purpose: str) -> bool:
    t = str(purpose or "").upper()
    if not t:
        return False
    has_div = "DIVIDEND" in t or "DIV." in t
    has_structural = any(k in t for k in ("SPLIT", "BONUS", "SUB-DIVISION",
                                          "SUB DIVISION", "RIGHTS", "CONSOLIDAT",
                                          "DEMERGER", "SCHEME OF ARRANGEMENT"))
    return has_div and not has_structural


# ----------------------------------------------------------- event detection

def implied_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each symbol/date, the factor implied by the bhavcopy's own PREV_CLOSE.

    On a normal session prev_close(t) == close(t-1) and the ratio is 1.0.
    On an ex-date NSE publishes the adjusted previous close, so the ratio is the
    corporate action factor. This needs no extra network call at all.
    """
    p = prices.sort_values(["symbol", "date"]).copy()
    p["date"] = pd.to_datetime(p["date"])
    prev_session_close = p.groupby("symbol", sort=False)["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p["prev_close"] / prev_session_close.replace(0, np.nan)
    out = p[["symbol", "date"]].copy()
    out["implied"] = ratio
    out["prev_session_close"] = prev_session_close
    return out


def detect_events(prices: pd.DataFrame,
                  corporate_actions: Optional[pd.DataFrame] = None
                  ) -> List[AdjustmentEvent]:
    """
    Build the list of adjustment events by combining both sources.

    `corporate_actions` is expected to have columns: symbol, ex_date, purpose.
    """
    imp = implied_factors(prices)
    imp = imp[imp["implied"].notna()]

    # candidate days where the exchange's own prev_close disagrees with yesterday
    cand = imp[(imp["implied"] - 1).abs() > MIN_EVENT_DEVIATION].copy()

    ca_lookup: Dict[Tuple[str, pd.Timestamp], Tuple[float, str]] = {}
    ca_rows: List[Tuple[str, pd.Timestamp, float, str]] = []
    if corporate_actions is not None and not corporate_actions.empty:
        ca = corporate_actions.copy()
        ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")
        ca = ca.dropna(subset=["ex_date", "symbol"])
        for r in ca.itertuples(index=False):
            factor, kind = parse_ratio(getattr(r, "purpose", ""))
            if factor is None or not np.isfinite(factor):
                continue
            if not (FACTOR_BOUNDS[0] <= factor <= FACTOR_BOUNDS[1]):
                continue
            key = (str(r.symbol).upper(), pd.Timestamp(r.ex_date).normalize())
            ca_lookup[key] = (factor, getattr(r, "purpose", ""))
            ca_rows.append((key[0], key[1], factor, getattr(r, "purpose", "")))

    events: List[AdjustmentEvent] = []
    seen: set = set()

    # 1) candidates found in the price data, corroborated where possible
    for r in cand.itertuples(index=False):
        sym = str(r.symbol).upper()
        d = pd.Timestamp(r.date).normalize()
        implied = float(r.implied)
        if not (FACTOR_BOUNDS[0] <= implied <= FACTOR_BOUNDS[1]):
            log.warning("%s %s: implied factor %.3f out of plausible range — ignored",
                        sym, d.date(), implied)
            continue
        parsed, purpose = (None, "")
        hit = ca_lookup.get((sym, d))
        if hit is None:                       # ex-date can land a session either way
            for off in (-1, 1):
                hit = ca_lookup.get((sym, d + pd.Timedelta(days=off)))
                if hit:
                    break
        if hit:
            parsed, purpose = hit[0], hit[1]

        if parsed is not None:
            source = "both"
            factor = parsed                    # a stated ratio beats an inferred one
            ev = AdjustmentEvent(sym, d, factor, source, purpose, implied, parsed)
            if ev.disagrees:
                log.warning(
                    "%s %s: corporate action says %.4f but prev_close implies %.4f "
                    "(purpose: %s). Using the stated ratio; verify this one by hand.",
                    sym, d.date(), parsed, implied, purpose[:70])
        else:
            # No filing on record. Only trust a clean, decisive ratio — this is
            # where a stale prev_close would otherwise inject a fake event.
            if abs(implied - 1) < 0.08:
                continue
            ev = AdjustmentEvent(sym, d, implied, "prev_close", "", implied, None)
        events.append(ev)
        seen.add((sym, d))

    # 2) filings with no visible price effect — usually the ex-date fell on a day
    #    we have no bar for. Apply them anyway, otherwise the gap survives.
    for sym, d, factor, purpose in ca_rows:
        if (sym, d) in seen:
            continue
        if abs(factor - 1) < MIN_EVENT_DEVIATION:
            continue
        events.append(AdjustmentEvent(sym, d, factor, "corporate_action",
                                      purpose, None, factor))

    events.sort(key=lambda e: (e.symbol, e.date))
    if events:
        both = sum(1 for e in events if e.source == "both")
        log.info("corporate actions: %d adjustment events across %d symbols "
                 "(%d confirmed by both sources)",
                 len(events), len({e.symbol for e in events}), both)
    return events


# ---------------------------------------------------------------- application

def apply_adjustments(prices: pd.DataFrame,
                      events: List[AdjustmentEvent]) -> pd.DataFrame:
    """
    Return `prices` with adjusted OHLC and volume.

    Convention: the MOST RECENT price is left untouched and history is scaled
    backwards. That way the entry price, stop and targets the site prints are
    always the real numbers you would see on your broker's terminal — an
    adjusted-forward series would quote prices that no longer exist.

    Adds columns: adj_open, adj_high, adj_low, adj_close, adj_volume, adj_factor.
    Leaves the raw columns untouched so nothing downstream is surprised.
    """
    p = prices.sort_values(["symbol", "date"]).copy()
    p["date"] = pd.to_datetime(p["date"])
    p["adj_factor"] = 1.0

    if events:
        by_symbol: Dict[str, List[AdjustmentEvent]] = {}
        for e in events:
            by_symbol.setdefault(e.symbol, []).append(e)

        idx_by_symbol = {s: g.index for s, g in p.groupby(p["symbol"].str.upper())}
        for sym, evs in by_symbol.items():
            idx = idx_by_symbol.get(sym)
            if idx is None or len(idx) == 0:
                continue
            dates = p.loc[idx, "date"].to_numpy()
            factors = np.ones(len(idx))
            for e in evs:
                # every bar strictly BEFORE the ex-date gets scaled
                factors[dates < np.datetime64(e.date)] *= e.factor
            p.loc[idx, "adj_factor"] = factors

    for c in ("open", "high", "low", "close"):
        p[f"adj_{c}"] = p[c] * p["adj_factor"]
    # share count moves inversely to price, so volume scales the other way
    p["adj_volume"] = p["volume"] / p["adj_factor"].replace(0, np.nan)
    return p


def adjusted_view(prices: pd.DataFrame,
                  corporate_actions: Optional[pd.DataFrame] = None,
                  enabled: bool = True) -> Tuple[pd.DataFrame, List[AdjustmentEvent]]:
    """
    One call for the pipeline: detect events, apply them, and swap the adjusted
    series into the ordinary OHLCV columns so every downstream indicator gets
    clean data without knowing this module exists.

    The raw prices survive as raw_open / raw_high / raw_low / raw_close, and the
    LAST bar is identical either way, so quoted entry prices stay real.
    """
    if not enabled:
        return prices, []

    events = detect_events(prices, corporate_actions)
    if not events:
        return prices, []

    adj = apply_adjustments(prices, events)
    for c in ("open", "high", "low", "close", "volume"):
        adj[f"raw_{c}"] = adj[c]
        adj[c] = adj[f"adj_{c}"]
    # turnover is price x quantity, so it is invariant — leave it alone.
    return adj, events


def events_frame(events: List[AdjustmentEvent]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["symbol", "date", "factor", "source",
                                     "purpose", "implied", "parsed", "disagrees"])
    return pd.DataFrame([{
        "symbol": e.symbol, "date": e.date, "factor": e.factor,
        "source": e.source, "purpose": e.purpose,
        "implied": e.implied, "parsed": e.parsed, "disagrees": e.disagrees,
    } for e in events])
