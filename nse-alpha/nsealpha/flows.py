"""
flows.py — institutional and derivatives positioning.

Price and fundamentals tell you what a stock is. Flow data tells you what large,
informed money is *doing about it*, and it is published free by NSE every evening
while almost nobody bothers to parse it.

Four sources, in descending order of usefulness:

1. F&O BHAVCOPY -> per-underlying positioning
   Aggregating every contract on a symbol gives:
     futures OI + change   The classic four-quadrant read. Price up on rising OI
                           is a long build-up (new money backing the move). Price
                           up on falling OI is short covering (a move with no new
                           conviction behind it, which typically stalls). Price
                           down on rising OI is a short build-up. Price down on
                           falling OI is long unwinding.
     put-call ratio (OI)   Above ~1.3 is heavy put writing, usually bullish
                           positioning; below ~0.6 is heavy call writing and
                           often marks a ceiling. Read at extremes, not in the
                           middle, and read as a contrary indicator with care.
     futures basis         Futures trading above spot (contango) shows willingness
                           to pay to be long; a persistent discount on a rising
                           stock is a warning that the move is cash-driven and
                           unsupported by leverage.
   Only ~180-220 stocks have derivatives, so this differentiates within the F&O
   universe and is simply absent elsewhere. The scoring layer handles that as
   missing coverage rather than as a zero.

2. PARTICIPANT-WISE OPEN INTEREST -> market regime
   NSE Clearing publishes how FIIs, DIIs, proprietary desks and retail clients are
   positioned across index futures and options. FII index-futures long ratio is
   the one worth watching: sustained readings under ~30% have historically
   coincided with defensive markets. This feeds the regime overlay, not stock
   scores.

3. FII / DII CASH ACTIVITY -> market regime
   Daily net buying in the cash segment. FIIs move the index; DIIs (domestic
   funds, driven by SIP inflows) increasingly absorb what FIIs sell. The
   divergence between them is more informative than either alone. Note this
   endpoint returns only the latest sessions with no archive, so history has to
   be accumulated by polling daily — start running it now and you have a series
   in three months.

4. SHAREHOLDING PATTERN -> per-stock institutional trend
   Quarterly, covers every listed company (not just F&O names), and shows the
   direction of FII, DII and promoter holding along with pledged shares. Slow-
   moving, but it is the only institutional-ownership signal available for the
   small and mid caps where it matters most.

A note on what is deliberately NOT here: AMFI monthly mutual fund portfolio
disclosures. They would be excellent — they show exactly what each fund bought
last month — but they are published per-AMC, in inconsistent Excel layouts, across
forty-odd asset managers, with no common schema. That is a scraping project in its
own right and it would break constantly. The shareholding pattern above gives a
coarser version of the same signal from one clean official source.
"""

from __future__ import annotations

import io
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .nseclient import NSEClient, NSEError, NotAvailable

log = logging.getLogger("nsealpha.flows")


# ------------------------------------------------------------- F&O bhavcopy

def parse_fno_bhavcopy(raw: bytes, d: date) -> pd.DataFrame:
    """
    Collapse every contract in the UDiFF F&O bhavcopy down to one row per
    underlying: futures OI and change, call/put OI and volume.

    FinInstrmTp codes: STF = stock future, IDF = index future,
                       STO = stock option, IDO = index option.
    """
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" not in df.columns:
        raise ValueError("unexpected F&O bhavcopy layout")

    num = lambda c: pd.to_numeric(df.get(c), errors="coerce")
    df["_sym"] = df["TckrSymb"].astype(str).str.strip().str.upper()
    df["_tp"] = df.get("FinInstrmTp", "").astype(str).str.strip().str.upper()
    df["_opt"] = df.get("OptnTp", "").astype(str).str.strip().str.upper()
    df["_oi"] = num("OpnIntrst")
    df["_doi"] = num("ChngInOpnIntrst")
    df["_vol"] = num("TtlTradgVol")
    df["_close"] = num("ClsPric")

    fut = df[df["_tp"].isin(["STF", "IDF"])]
    opt = df[df["_tp"].isin(["STO", "IDO"])]

    # Near-month futures carry the signal; far months are mostly rollover noise.
    if "XpryDt" in df.columns and not fut.empty:
        fut = fut.copy()
        fut["_exp"] = pd.to_datetime(fut["XpryDt"], errors="coerce")
        near = fut.groupby("_sym")["_exp"].transform("min")
        fut = fut[fut["_exp"] == near]

    fut_agg = fut.groupby("_sym").agg(
        fut_close=("_close", "mean"),
        fut_oi=("_oi", "sum"),
        fut_oi_chg=("_doi", "sum"),
        fut_volume=("_vol", "sum")) if not fut.empty else pd.DataFrame()

    calls = opt[opt["_opt"] == "CE"].groupby("_sym").agg(
        call_oi=("_oi", "sum"), call_volume=("_vol", "sum")) \
        if not opt.empty else pd.DataFrame()
    puts = opt[opt["_opt"] == "PE"].groupby("_sym").agg(
        put_oi=("_oi", "sum"), put_volume=("_vol", "sum")) \
        if not opt.empty else pd.DataFrame()

    out = pd.concat([fut_agg, calls, puts], axis=1)
    if out.empty:
        return pd.DataFrame()
    out.index.name = "symbol"
    out = out.reset_index()
    out["date"] = pd.Timestamp(d)
    return out


def backfill_fno(client: NSEClient, store, days_back: int = 400,
                 progress=None) -> int:
    have = store.fno_dates()
    end = date.today()
    start = end - timedelta(days=days_back)
    days = [d for d in _weekdays(start, end) if d.strftime("%Y-%m-%d") not in have]
    days.sort(reverse=True)

    ok = miss = errors = 0
    first_error = None
    for i, d in enumerate(days):
        try:
            df = parse_fno_bhavcopy(client.bhavcopy_fno(d), d)
        except (NotAvailable, ValueError):
            miss += 1
            continue
        except NSEError as e:
            errors += 1
            if first_error is None:
                first_error = str(e)
                log.warning("F&O bhavcopy download failing: %s", e)
            continue
        if not df.empty:
            store.upsert_fno(df)
            ok += 1
        if progress and i % 10 == 0:
            progress(i + 1, len(days), f"F&O {d}")
    log.info("F&O bhavcopy: %d days added, %d unavailable, %d failed", ok, miss, errors)
    return ok


def _weekdays(start: date, end: date) -> List[date]:
    out, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            out.append(cur)
        cur += timedelta(days=1)
    return out


# ---------------------------------------------------- participant-wise OI

def parse_participant_oi(raw: bytes, d: date) -> pd.DataFrame:
    """
    fao_participant_oi_DDMMYYYY.csv

    The file carries a title line before the header, then one row per client type
    (Client, DII, FII, Pro) with long/short OI across index and stock derivatives.
    Returns tidy (date, metric, value) rows for the market_flows table.
    """
    text = raw.decode("utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    # find the real header row — the one naming the client type column
    hdr = 0
    for i, l in enumerate(lines[:6]):
        if "client type" in l.lower():
            hdr = i
            break
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [c.strip() for c in df.columns]
    key = next((c for c in df.columns if "client" in c.lower()), df.columns[0])
    df[key] = df[key].astype(str).str.strip().str.upper()

    rows = []
    for _, row in df.iterrows():
        who = str(row[key]).upper()
        if who not in ("CLIENT", "DII", "FII", "PRO", "TOTAL"):
            continue
        for col in df.columns:
            if col == key:
                continue
            val = pd.to_numeric(str(row[col]).replace(",", ""), errors="coerce")
            if pd.isna(val):
                continue
            metric = f"{who}_{col.strip().replace(' ', '_').lower()}"
            rows.append({"date": pd.Timestamp(d), "metric": metric, "value": float(val)})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Derive the one number actually worth looking at: FII net position in index
    # futures as a share of their total index-futures OI.
    def _get(m):
        hit = out[out["metric"] == m]["value"]
        return float(hit.iloc[0]) if len(hit) else np.nan

    fl = _get("FII_future_index_long")
    fs = _get("FII_future_index_short")
    if np.isfinite(fl) and np.isfinite(fs) and (fl + fs) > 0:
        out = pd.concat([out, pd.DataFrame([{
            "date": pd.Timestamp(d), "metric": "FII_index_fut_long_pct",
            "value": 100.0 * fl / (fl + fs)}])], ignore_index=True)
    return out


def ingest_participant_oi(client: NSEClient, store, days_back: int = 120) -> int:
    have = set(store.load_market_flows()["date"].dt.strftime("%Y-%m-%d")) \
        if not store.load_market_flows().empty else set()
    total = 0
    for d in sorted(_weekdays(date.today() - timedelta(days=days_back), date.today()),
                    reverse=True):
        if d.strftime("%Y-%m-%d") in have:
            continue
        try:
            df = parse_participant_oi(client.participant_oi(d), d)
        except (NotAvailable, NSEError, ValueError, KeyError):
            continue
        if not df.empty:
            total += store.upsert_market_flows(df)
    return total


# ------------------------------------------------------------- FII / DII cash

def ingest_fii_dii(client: NSEClient, store) -> int:
    """
    Latest FII/DII cash-market activity. The endpoint has no archive, so this is
    append-only: run it daily and the series accumulates.
    """
    try:
        payload = client.fii_dii_activity()
    except (NSEError, NotAvailable) as e:
        log.warning("FII/DII activity unavailable: %s",
                    str(e).split("(Caused by")[0].strip())
        return 0

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    recs = []
    for r in rows or []:
        cat = str(r.get("category", "")).upper()
        who = "FII" if "FII" in cat or "FPI" in cat else ("DII" if "DII" in cat else None)
        if who is None:
            continue
        try:
            d = pd.to_datetime(r.get("date"), dayfirst=True)
        except Exception:
            continue
        for src, name in (("buyValue", "buy"), ("sellValue", "sell"),
                          ("netValue", "net")):
            v = pd.to_numeric(str(r.get(src, "")).replace(",", ""), errors="coerce")
            if pd.notna(v):
                recs.append({"date": d, "metric": f"{who}_{name}_cr", "value": float(v)})
    if not recs:
        return 0
    return store.upsert_market_flows(pd.DataFrame(recs))


# ------------------------------------------------------------- shareholding

def ingest_shareholding(client: NSEClient, store, symbols: List[str],
                        progress=None) -> int:
    """
    Quarterly shareholding pattern per company. One request per symbol, so this
    runs on the filtered universe only and is refreshed monthly, not daily.
    """
    done = 0
    for i, sym in enumerate(symbols):
        try:
            payload = client.shareholding(sym)
        except (NSEError, NotAvailable):
            continue
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        recs = []
        for r in rows or []:
            period = str(r.get("date") or r.get("asOnDate") or "").strip()
            if not period:
                continue

            def num(*keys):
                for k in keys:
                    v = r.get(k)
                    if v not in (None, "", "-"):
                        x = pd.to_numeric(str(v).replace(",", "").replace("%", ""),
                                          errors="coerce")
                        if pd.notna(x):
                            return float(x)
                return np.nan

            recs.append({
                "symbol": sym, "period": period,
                "promoter": num("promoter", "promoterAndPromoterGroup", "pr_and_prgrp"),
                "pledge": num("pledge", "pledged", "encumbered"),
                "fii": num("fii", "foreignInstitutions", "fpi"),
                "dii": num("dii", "domesticInstitutions", "mf"),
                "public": num("public", "publicShareholding"),
            })
        if recs:
            done += store.upsert_shareholding(pd.DataFrame(recs))
        if progress and i % 25 == 0:
            progress(i + 1, len(symbols), f"holding {sym}")
    return done


def shareholding_trend(store) -> pd.DataFrame:
    """
    Latest values plus quarter-on-quarter change per symbol — the change is the
    part that carries information.
    """
    sh = store.load_shareholding()
    if sh.empty:
        return pd.DataFrame()
    sh = sh.copy()
    sh["_ord"] = pd.to_datetime(sh["period"], errors="coerce", dayfirst=True)
    sh = sh.dropna(subset=["_ord"]).sort_values(["symbol", "_ord"])
    g = sh.groupby("symbol")
    last = g.tail(1).set_index("symbol")
    prev = g.nth(-2)
    prev = prev.set_index("symbol") if "symbol" in prev.columns else prev
    out = pd.DataFrame(index=last.index)
    for col in ("promoter", "pledge", "fii", "dii"):
        out[col] = last[col]
        if col in getattr(prev, "columns", []):
            out[f"{col}_chg"] = last[col] - prev[col].reindex(last.index)
        else:
            out[f"{col}_chg"] = np.nan
    return out


# --------------------------------------------------------- derived per-stock

def fno_features(store, as_of: pd.Timestamp, lookback_days: int = 40) -> pd.DataFrame:
    """
    Per-symbol derivatives features as of a date: OI change over 1 and 5 sessions,
    put-call ratio and its trend, and where OI sits within its recent range.
    """
    since = (as_of - pd.Timedelta(days=lookback_days * 2)).strftime("%Y-%m-%d")
    fno = store.load_fno(since=since)
    if fno.empty:
        return pd.DataFrame()
    fno = fno[fno["date"] <= as_of].sort_values(["symbol", "date"])

    rows = {}
    for sym, g in fno.groupby("symbol", sort=False):
        if len(g) < 3:
            continue
        last = g.iloc[-1]
        oi = g["fut_oi"]
        oi_1d = float(oi.iloc[-1] / oi.iloc[-2] - 1) * 100 if len(oi) > 1 and oi.iloc[-2] else np.nan
        oi_5d = float(oi.iloc[-1] / oi.iloc[-6] - 1) * 100 if len(oi) > 6 and oi.iloc[-6] else np.nan
        oi_pctile = float((oi.tail(40) <= oi.iloc[-1]).mean() * 100) if len(oi) >= 10 else np.nan

        pcr = (g["put_oi"] / g["call_oi"].replace(0, np.nan))
        pcr_now = float(pcr.iloc[-1]) if pd.notna(pcr.iloc[-1]) else np.nan
        pcr_5d = float(pcr.tail(5).mean()) if pcr.notna().any() else np.nan
        pcr_trend = (pcr_now - pcr_5d) if np.isfinite(pcr_now) and np.isfinite(pcr_5d) else np.nan

        rows[sym] = {
            "fut_oi": float(last["fut_oi"]) if pd.notna(last["fut_oi"]) else np.nan,
            "fut_close": float(last["fut_close"]) if pd.notna(last["fut_close"]) else np.nan,
            "oi_chg_1d_pct": oi_1d,
            "oi_chg_5d_pct": oi_5d,
            "oi_percentile_40d": oi_pctile,
            "pcr_oi": pcr_now,
            "pcr_trend": pcr_trend,
            "opt_volume": float((last.get("call_volume") or 0) + (last.get("put_volume") or 0)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def market_flow_summary(store, as_of: pd.Timestamp) -> Dict[str, float]:
    """Market-level flow numbers for the regime overlay."""
    mf = store.load_market_flows(
        since=(as_of - pd.Timedelta(days=45)).strftime("%Y-%m-%d"))
    if mf.empty:
        return {}
    mf = mf[mf["date"] <= as_of]
    out: Dict[str, float] = {}

    def series(metric):
        s = mf[mf["metric"] == metric].sort_values("date")["value"]
        return s

    for who in ("FII", "DII"):
        s = series(f"{who}_net_cr")
        if len(s):
            out[f"{who.lower()}_net_latest_cr"] = float(s.iloc[-1])
            out[f"{who.lower()}_net_5d_cr"] = float(s.tail(5).sum())
            out[f"{who.lower()}_net_20d_cr"] = float(s.tail(20).sum())

    s = series("FII_index_fut_long_pct")
    if len(s):
        out["fii_index_fut_long_pct"] = float(s.iloc[-1])
        out["fii_index_fut_long_pct_5d"] = float(s.tail(5).mean())
    return out
