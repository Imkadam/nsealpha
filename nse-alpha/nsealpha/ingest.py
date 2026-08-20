"""
ingest.py — turn NSE's files and endpoints into tidy DataFrames, then into SQLite.

Design note: we build history from *bhavcopies*, not per-symbol history calls.
One bhavcopy = the entire market for one day. Backfilling two years costs ~500
requests instead of ~2000 (one per stock), and it is the only practical way to
cover "all NSE listed" without getting rate-limited into oblivion.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .nseclient import NSEClient, NotAvailable, NSEError

log = logging.getLogger("nsealpha.ingest")


# ------------------------------------------------------------------ bhavcopies

def parse_bhavcopy_full(raw: bytes, d: date) -> pd.DataFrame:
    """
    sec_bhavdata_full_DDMMYYYY.csv

    Real NSE header (note the leading spaces, they are in the file):
      SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
      LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,
      NO_OF_TRADES, DELIV_QTY, DELIV_PER
    """
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df.columns:
        raise ValueError("unexpected bhavcopy layout")

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    num = ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "LAST_PRICE",
           "CLOSE_PRICE", "AVG_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS",
           "NO_OF_TRADES", "DELIV_QTY", "DELIV_PER"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"-": np.nan, "": np.nan}), errors="coerce")

    out = pd.DataFrame({
        "symbol": df["SYMBOL"],
        "date": pd.Timestamp(d),
        "series": df.get("SERIES"),
        "open": df.get("OPEN_PRICE"),
        "high": df.get("HIGH_PRICE"),
        "low": df.get("LOW_PRICE"),
        "close": df.get("CLOSE_PRICE"),
        "prev_close": df.get("PREV_CLOSE"),
        "last_price": df.get("LAST_PRICE"),
        "volume": df.get("TTL_TRD_QNTY"),
        # TURNOVER_LACS is in lakhs of rupees -> convert to rupees
        "turnover": df.get("TURNOVER_LACS") * 1e5 if "TURNOVER_LACS" in df else np.nan,
        "trades": df.get("NO_OF_TRADES"),
        "deliv_qty": df.get("DELIV_QTY"),
        "deliv_pct": df.get("DELIV_PER"),
    })
    return out


def parse_bhavcopy_udiff(raw: bytes, d: date) -> pd.DataFrame:
    """UDiFF format (post 8-Jul-2024) or the legacy cmDDMONYYYYbhav.csv layout."""
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    if "TckrSymb" in df.columns:                       # UDiFF
        df = df[df.get("FinInstrmTp", "STK").astype(str).str.upper().isin(["STK", "EQ"])]
        out = pd.DataFrame({
            "symbol": df["TckrSymb"].astype(str).str.strip(),
            "date": pd.Timestamp(d),
            "series": df.get("SctySrs"),
            "open": pd.to_numeric(df.get("OpnPric"), errors="coerce"),
            "high": pd.to_numeric(df.get("HghPric"), errors="coerce"),
            "low": pd.to_numeric(df.get("LwPric"), errors="coerce"),
            "close": pd.to_numeric(df.get("ClsPric"), errors="coerce"),
            "prev_close": pd.to_numeric(df.get("PrvsClsgPric"), errors="coerce"),
            "last_price": pd.to_numeric(df.get("LastPric"), errors="coerce"),
            "volume": pd.to_numeric(df.get("TtlTradgVol"), errors="coerce"),
            "turnover": pd.to_numeric(df.get("TtlTrfVal"), errors="coerce"),
            "trades": pd.to_numeric(df.get("TtlNbOfTxsExctd"), errors="coerce"),
            "deliv_qty": np.nan, "deliv_pct": np.nan,
        })
    else:                                              # legacy
        df.columns = [c.upper() for c in df.columns]
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "date": pd.Timestamp(d),
            "series": df.get("SERIES"),
            "open": pd.to_numeric(df.get("OPEN"), errors="coerce"),
            "high": pd.to_numeric(df.get("HIGH"), errors="coerce"),
            "low": pd.to_numeric(df.get("LOW"), errors="coerce"),
            "close": pd.to_numeric(df.get("CLOSE"), errors="coerce"),
            "prev_close": pd.to_numeric(df.get("PREVCLOSE"), errors="coerce"),
            "last_price": pd.to_numeric(df.get("LAST"), errors="coerce"),
            "volume": pd.to_numeric(df.get("TOTTRDQTY"), errors="coerce"),
            "turnover": pd.to_numeric(df.get("TOTTRDVAL"), errors="coerce"),
            "trades": pd.to_numeric(df.get("TOTALTRADES"), errors="coerce"),
            "deliv_qty": np.nan, "deliv_pct": np.nan,
        })
    return out


def parse_index_close(raw: bytes, d: date) -> pd.DataFrame:
    """
    ind_close_all_DDMMYYYY.csv
    Header: Index Name, Index Date, Open Index Value, High Index Value,
            Low Index Value, Closing Index Value, Points Change, Change(%),
            Volume, Turnover (Rs. Cr.), P/E, P/B, Div Yield
    """
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    name_col = next((c for c in df.columns if "Index Name" in c), df.columns[0])
    close_col = next((c for c in df.columns if "Closing" in c), None)
    out = pd.DataFrame({
        "index_name": df[name_col].astype(str).str.strip(),
        "date": pd.Timestamp(d),
        "open": pd.to_numeric(df.get("Open Index Value"), errors="coerce"),
        "high": pd.to_numeric(df.get("High Index Value"), errors="coerce"),
        "low": pd.to_numeric(df.get("Low Index Value"), errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce") if close_col else np.nan,
        "pct_change": pd.to_numeric(df.get("Change(%)"), errors="coerce"),
    })
    return out.dropna(subset=["close"])


# -------------------------------------------------------------- trading-day walk

def trading_days(start: date, end: date) -> List[date]:
    """Weekdays between start and end. NSE holidays simply 404 and are skipped."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def backfill_prices(client: NSEClient, store, days_back: int = 900,
                    progress=None) -> int:
    """
    Walk backwards from today, downloading any bhavcopy we don't already have.
    Holidays/missing files are skipped silently. Cached files cost nothing.
    """
    have = store.price_dates()
    end = date.today()
    start = end - timedelta(days=days_back)
    wanted = [d for d in trading_days(start, end) if d.strftime("%Y-%m-%d") not in have]
    wanted.sort(reverse=True)          # newest first, so a partial run is still useful

    ok = miss = errors = 0
    first_error = None
    for i, d in enumerate(wanted):
        try:
            raw = client.bhavcopy_full(d)
            df = parse_bhavcopy_full(raw, d)
        except (NotAvailable, ValueError):
            # holiday, or NSE hasn't published yet — try the UDiFF file instead
            try:
                raw = client.bhavcopy_udiff(d)
                df = parse_bhavcopy_udiff(raw, d)
            except (NotAvailable, ValueError, NSEError):
                miss += 1
                continue
        except NSEError as e:
            # Connection-level failures usually all share one cause (proxy, 403,
            # no network). Report it once instead of once per trading day.
            errors += 1
            if first_error is None:
                first_error = str(e)
                log.warning("bhavcopy download failing: %s", e)
            elif errors == 4:
                log.warning("...suppressing further download errors; "
                            "a summary follows at the end")
            continue
        store.upsert_prices(df)
        ok += 1
        if progress and i % 10 == 0:
            progress(i + 1, len(wanted), d)

    if errors:
        log.error("%d of %d bhavcopy downloads failed. First error: %s",
                  errors, len(wanted), first_error)
        if ok == 0:
            log.error("Nothing downloaded at all. Common causes: no internet, a "
                      "corporate proxy, or NSE blocking this IP. Try setting "
                      "data.server_mode: true in config.yaml, lowering "
                      "requests_per_second, or running from a home connection.")
    log.info("bhavcopy backfill: %d days added, %d unavailable (holidays), %d failed",
             ok, miss, errors)
    return ok


def backfill_indices(client: NSEClient, store, days_back: int = 900) -> int:
    have = {r for r in _index_dates(store)}
    end = date.today()
    start = end - timedelta(days=days_back)
    wanted = [d for d in trading_days(start, end) if d.strftime("%Y-%m-%d") not in have]
    wanted.sort(reverse=True)
    ok = 0
    for d in wanted:
        try:
            df = parse_index_close(client.index_close_all(d), d)
        except (NotAvailable, NSEError, ValueError):
            continue
        store.upsert_indices(df)
        ok += 1
    return ok


def _index_dates(store) -> set:
    return {r[0] for r in store.conn.execute("SELECT DISTINCT date FROM indices").fetchall()}


# ------------------------------------------------------------ security master

SURVEILLANCE_SERIES = {"BE", "BZ", "IL", "XT", "XC", "XD"}   # trade-to-trade / restricted


def build_security_master(client: NSEClient, store, prices: pd.DataFrame) -> int:
    """
    Best-effort security master. NSE's sec_list gives surveillance flags; the
    quote endpoint gives sector/industry but costs one call per symbol, so we
    only fill sector for names that survive the liquidity filter (see universe.py).
    """
    latest = prices.groupby("symbol").tail(1)
    rows = pd.DataFrame({
        "symbol": latest["symbol"],
        "series": latest["series"],
        "surveillance": latest["series"].apply(
            lambda s: "TRADE_TO_TRADE" if str(s).upper() in SURVEILLANCE_SERIES else ""),
        "updated": date.today().isoformat(),
    })
    for d in (date.today(), date.today() - timedelta(days=1),
              date.today() - timedelta(days=2)):
        try:
            sl = pd.read_csv(io.BytesIO(client.security_list(d)))
            sl.columns = [c.strip().upper() for c in sl.columns]
            symcol = next((c for c in sl.columns if "SYMBOL" in c), None)
            if symcol:
                sl = sl.rename(columns={symcol: "symbol"})
                keep = {"symbol"}
                for src, dst in (("NAME OF COMPANY", "name"), ("SECURITY NAME", "name"),
                                 (" ISIN NUMBER", "isin"), ("ISIN NUMBER", "isin"),
                                 ("DATE OF LISTING", "listing_date"),
                                 ("FACE VALUE", "face_value")):
                    if src in sl.columns:
                        sl = sl.rename(columns={src: dst})
                        keep.add(dst)
                sl["symbol"] = sl["symbol"].astype(str).str.strip()
                rows = rows.merge(sl[list(keep)], on="symbol", how="left")
            break
        except Exception:
            continue
    return store.upsert_securities(rows)


# ------------------------------------------------------- announcements / news

POSITIVE_LEXICON = {
    r"\border(s)? (win|won|received|bagged|secured)": 3.0,
    r"\bnew order\b|\border book\b|\bletter of (award|intent)\b|\bLOA\b": 2.5,
    r"\bbuy ?back\b": 2.5,
    r"\bbonus (issue|share)": 2.0,
    r"\b(stock )?split\b": 1.0,
    r"\bdividend\b": 1.2,
    r"\brecord (date|profit|revenue)": 1.5,
    r"\bhighest ever\b|\ball[- ]time high\b": 2.5,
    r"\bcapacity expansion\b|\bcapex\b|\bnew plant\b|\bcommission(ed|ing)\b": 2.0,
    r"\bacquisition\b|\bacquir(e|ed|ing)\b|\bmerger\b|\bstake (purchase|acquisition)": 1.5,
    r"\bjoint venture\b|\bstrategic (partnership|alliance)\b|\bMoU\b": 1.8,
    r"\bcredit rating (upgrade|upgraded)\b|\brating upgrade\b": 2.5,
    r"\bpreferential (issue|allotment)\b|\bfund rais": 0.8,
    r"\bdebt (reduction|repayment|prepay)": 2.0,
    r"\bapproval (received|granted)\b|\bUSFDA approval\b|\bpatent (granted|approval)": 2.5,
    r"\bQIP\b": 0.5,
    r"\bpromoter (buying|acquisition|increase)": 2.5,
}

NEGATIVE_LEXICON = {
    r"\bresignation\b|\bresign(ed|s)\b|\bcessation\b": -2.0,
    r"\b(CEO|CFO|MD|managing director|auditor) (resign|exit|step)": -3.0,
    r"\bauditor (qualification|qualified opinion|adverse)": -4.0,
    r"\bdefault\b|\bNPA\b|\binsolvency\b|\bIBC\b|\bNCLT\b": -4.5,
    r"\bpledge(d)?\b|\binvocation of pledge\b": -2.5,
    r"\brating (downgrade|downgraded)\b|\bdowngrade\b": -3.0,
    r"\bSEBI (order|penalty|show cause|notice)\b|\bpenalt(y|ies)\b|\bfine\b": -2.5,
    r"\bsearch and seizure\b|\bincome tax (raid|search)\b|\bED (raid|summons)\b": -4.0,
    r"\bfraud\b|\bmisappropriation\b|\bembezzle": -5.0,
    r"\bplant (shutdown|closure)\b|\bstrike\b|\blockout\b|\bfire (incident|accident)\b": -2.5,
    r"\bloss\b(?!.*reduc)": -1.5,
    r"\bASM\b|\bGSM\b|\bsurveillance\b": -2.0,
    r"\bdelisting\b|\bsuspension of trading\b": -3.5,
    r"\bpromoter (selling|sale|stake sale|offloaded)": -2.5,
    r"\border cancel|\bcontract terminat": -3.0,
}


def score_announcement(text: str) -> tuple[float, str]:
    """Keyword-lexicon sentiment. Crude, transparent, and surprisingly hard to beat
    on Indian corporate filings, which use very formulaic language."""
    if not text:
        return 0.0, ""
    t = str(text)
    score, tags = 0.0, []
    for pat, w in POSITIVE_LEXICON.items():
        if re.search(pat, t, flags=re.I):
            score += w
            tags.append(pat.split("\\b")[1][:18] if "\\b" in pat else pat[:18])
    for pat, w in NEGATIVE_LEXICON.items():
        if re.search(pat, t, flags=re.I):
            score += w
            tags.append(pat.split("\\b")[1][:18] if "\\b" in pat else pat[:18])
    return float(np.clip(score, -6, 6)), ",".join(tags[:4])


def ingest_announcements(client: NSEClient, store, days_back: int = 45) -> int:
    end, total = date.today(), 0
    start = end - timedelta(days=days_back)
    # NSE caps the announcement window; walk it in ~14-day chunks
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=14), end)
        try:
            rows = client.announcements(cur, chunk_end)
        except (NSEError, NotAvailable) as e:
            log.warning("announcements %s..%s unavailable (%s)",
                        cur, chunk_end, str(e).split("(Caused by")[0].strip())
            cur = chunk_end + timedelta(days=1)
            continue
        recs = []
        for r in rows or []:
            sym = (r.get("symbol") or r.get("sym") or "").strip().upper()
            if not sym:
                continue
            subj = r.get("desc") or r.get("subject") or ""
            detail = r.get("attchmntText") or r.get("smIndustry") or ""
            dt = r.get("an_dt") or r.get("sort_date") or r.get("exchdisstime") or ""
            dt = str(dt)[:10].replace("/", "-")
            try:
                dt = pd.to_datetime(dt, dayfirst=("-" in dt and len(dt.split("-")[0]) == 2)
                                    ).strftime("%Y-%m-%d")
            except Exception:
                dt = chunk_end.isoformat()
            s, tags = score_announcement(f"{subj} {detail}")
            recs.append({"symbol": sym, "date": dt, "subject": str(subj)[:300],
                         "detail": str(detail)[:600], "score": s, "tags": tags})
        if recs:
            total += store.upsert_announcements(pd.DataFrame(recs))
        cur = chunk_end + timedelta(days=1)
    return total


def ingest_corporate_actions(client: NSEClient, store, days_back: int = 900) -> int:
    """
    Splits, bonuses, rights and dividends with their ex-dates. This is what makes
    the price-adjustment in adjust.py trustworthy rather than merely plausible.
    NSE caps the window per request, so walk it in 90-day chunks.
    """
    end = date.today() + timedelta(days=30)      # include announced future ex-dates
    start = end - timedelta(days=days_back + 30)
    total, cur = 0, start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=90), end)
        try:
            rows = client.corporate_actions(cur, chunk_end)
        except (NSEError, NotAvailable) as e:
            log.warning("corporate actions %s..%s unavailable (%s)",
                        cur, chunk_end, str(e).split("(Caused by")[0].strip())
            cur = chunk_end + timedelta(days=1)
            continue
        recs = []
        for r in rows or []:
            sym = (r.get("symbol") or "").strip().upper()
            purpose = r.get("subject") or r.get("purpose") or ""
            ex = r.get("exDate") or r.get("ex_date") or ""
            if not sym or not ex:
                continue
            try:
                ex_norm = pd.to_datetime(str(ex)[:11], dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                continue
            rec_date = r.get("recDate") or r.get("record_date") or ""
            recs.append({"symbol": sym, "ex_date": ex_norm,
                         "purpose": str(purpose)[:300], "record_date": str(rec_date)[:11]})
        if recs:
            total += store.upsert_corporate_actions(pd.DataFrame(recs))
        cur = chunk_end + timedelta(days=1)
    return total


def ingest_deals(client: NSEClient, store, days_back: int = 45) -> int:
    end = date.today()
    start = end - timedelta(days=days_back)
    try:
        payload = client.bulk_block_deals(start, end)
    except (NSEError, NotAvailable) as e:
        log.warning("bulk/block deals failed: %s", e)
        return 0
    rows = payload.get("data", payload if isinstance(payload, list) else [])
    recs = []
    for r in rows or []:
        sym = (r.get("BD_SYMBOL") or r.get("symbol") or "").strip().upper()
        if not sym:
            continue
        dt = str(r.get("BD_DT_DATE") or r.get("date") or "")[:10]
        try:
            dt = pd.to_datetime(dt, dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            continue
        side = str(r.get("BD_BUY_SELL") or r.get("buySell") or "").upper()[:4]
        recs.append({
            "symbol": sym, "date": dt,
            "client": str(r.get("BD_CLIENT_NAME") or "")[:120],
            "side": "BUY" if side.startswith("B") else "SELL",
            "qty": pd.to_numeric(r.get("BD_QTY_TRD"), errors="coerce"),
            "price": pd.to_numeric(r.get("BD_TP_WATP"), errors="coerce"),
            "kind": "BULK",
        })
    return store.upsert_deals(pd.DataFrame(recs)) if recs else 0


# ------------------------------------------------------------------ fundamentals

def _safe(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in ("", "-", None):
            return d[k]
    return default


def fetch_fundamentals_nse(client: NSEClient, symbol: str) -> dict:
    """
    What NSE gives us directly from /quote-equity:
      metadata.pdSymbolPe  -> trailing P/E for the symbol
      metadata.pdSectorPe  -> P/E of its sector (relative valuation, free)
      metadata.industry    -> sector/industry classification
      securityInfo / priceInfo -> issued size, 52w range, VWAP, face value
    Market cap = issuedSize * close.
    """
    q = client.quote(symbol)
    meta = q.get("metadata", {}) or {}
    info = q.get("info", {}) or {}
    price = q.get("priceInfo", {}) or {}
    sec = q.get("securityInfo", {}) or {}
    ind = q.get("industryInfo", {}) or {}

    def num(x):
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return np.nan

    close = num(_safe(price, "lastPrice", "close"))
    issued = num(_safe(sec, "issuedSize"))
    return {
        "source": "nse",
        "company": _safe(info, "companyName", default=symbol),
        "sector": _safe(ind, "macro", "sector", default=_safe(meta, "industry")),
        "industry": _safe(ind, "industry", "basicIndustry", default=_safe(meta, "industry")),
        "pe": num(_safe(meta, "pdSymbolPe")),
        "sector_pe": num(_safe(meta, "pdSectorPe")),
        "market_cap_cr": (close * issued / 1e7) if close and issued else np.nan,
        "week52_high": num(_safe(price.get("weekHighLow", {}), "max")),
        "week52_low": num(_safe(price.get("weekHighLow", {}), "min")),
        "face_value": num(_safe(sec, "faceValue")),
        "listing_date": _safe(meta, "listingDate"),
    }


def fetch_fundamentals_yf(symbol: str) -> dict:
    """
    Fallback / enrichment via yfinance. NSE does not publish ROE, D/E, margins or
    P/B through its public JSON, so if the user has yfinance installed we top up
    the quality and growth ratios from there. Entirely optional.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}
    try:
        t = yf.Ticker(f"{symbol}.NS")
        i = t.info or {}
    except Exception:
        return {}
    if not i:
        return {}
    g = lambda *k: next((i[x] for x in k if i.get(x) is not None), np.nan)
    return {
        "source_extra": "yfinance",
        "pb": g("priceToBook"),
        "peg": g("pegRatio", "trailingPegRatio"),
        "roe": (g("returnOnEquity") or np.nan) * 100 if i.get("returnOnEquity") else np.nan,
        "roa": (g("returnOnAssets") or np.nan) * 100 if i.get("returnOnAssets") else np.nan,
        "debt_to_equity": g("debtToEquity"),
        "current_ratio": g("currentRatio"),
        "operating_margin": (g("operatingMargins") or np.nan) * 100 if i.get("operatingMargins") else np.nan,
        "profit_margin": (g("profitMargins") or np.nan) * 100 if i.get("profitMargins") else np.nan,
        "revenue_growth": (g("revenueGrowth") or np.nan) * 100 if i.get("revenueGrowth") else np.nan,
        "earnings_growth": (g("earningsGrowth", "earningsQuarterlyGrowth") or np.nan) * 100
                            if (i.get("earningsGrowth") or i.get("earningsQuarterlyGrowth")) else np.nan,
        "dividend_yield": (g("dividendYield") or np.nan) * 100 if i.get("dividendYield") else np.nan,
        "ev_to_ebitda": g("enterpriseToEbitda"),
        "beta": g("beta"),
        "held_by_insiders": (g("heldPercentInsiders") or np.nan) * 100 if i.get("heldPercentInsiders") else np.nan,
        "held_by_institutions": (g("heldPercentInstitutions") or np.nan) * 100 if i.get("heldPercentInstitutions") else np.nan,
        "market_cap_cr_yf": (g("marketCap") or np.nan) / 1e7 if i.get("marketCap") else np.nan,
    }


def refresh_fundamentals(client: NSEClient, store, symbols: Iterable[str],
                         max_age_days: int = 7, use_yf: bool = True,
                         progress=None) -> int:
    """Refresh fundamentals for the (already filtered) universe. Weekly cadence."""
    today = date.today().isoformat()
    symbols = list(symbols)
    done = 0
    for i, sym in enumerate(symbols):
        age = store.fundamentals_age_days(sym)
        if age is not None and age < max_age_days:
            continue
        payload = {}
        try:
            payload.update(fetch_fundamentals_nse(client, sym))
        except Exception as e:
            log.debug("nse fundamentals %s: %s", sym, e)
        if use_yf:
            try:
                payload.update({k: v for k, v in fetch_fundamentals_yf(sym).items()
                                if v is not None})
            except Exception as e:
                log.debug("yf fundamentals %s: %s", sym, e)
        if payload:
            store.upsert_fundamentals(sym, today, payload)
            done += 1
        if progress and i % 25 == 0:
            progress(i + 1, len(symbols), sym)
    return done
