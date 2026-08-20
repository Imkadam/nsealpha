"""
nseclient.py — a polite, resilient HTTP client for nseindia.com.

NSE does not publish an official public API. What exists are the JSON endpoints
its own website calls, plus a set of static CSV/ZIP archives. Both work, but the
JSON endpoints require a real browser-ish session: you must first hit an HTML
page to collect the `nsit`/`nseappid` cookies, send a plausible User-Agent and a
Referer, and keep the request rate low. This module handles all of that.

Endpoint map (verified against NSE's current site structure):

  Archives (nsearchives.nseindia.com) — no cookie needed, but same headers help:
    /products/content/sec_bhavdata_full_DDMMYYYY.csv
        Full daily bhavcopy for every equity: OHLC, volume, turnover, trades,
        AND delivery quantity / delivery percentage. This is the single most
        useful file NSE publishes and it is one request for the whole market.
    /content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
        UDiFF-format bhavcopy (the post-8-July-2024 format). Used as a fallback
        when sec_bhavdata_full is missing.
    /content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
        Legacy bhavcopy format, for dates before 8 July 2024.
    /content/indices/ind_close_all_DDMMYYYY.csv
        Closing values for every NSE index — our benchmark + sector series.
    /content/equities/sec_list_DDMMYYYY.csv
        Security list including surveillance (ASM/GSM) flags.

  JSON API (www.nseindia.com/api) — cookie required:
    /equity-stock-indices?index=NIFTY%20500     index constituents
    /allIndices                                 all index levels
    /corporate-announcements?index=equities     corporate filings feed
    /corporates-corporateActions?index=equities dividends/splits/bonus
    /corporates-financial-results?index=equities&period=Quarterly
    /corporate-share-holdings-master?symbol=X   promoter / FII / DII holding
    /quote-equity?symbol=X                      live quote + P/E + sector P/E
    /historicalOR/bulk-block-short-deals        bulk & block deals
    /marketStatus, /holiday-master
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import random
import shutil
import threading
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("nsealpha.client")

BASE = "https://www.nseindia.com/api"
ARCHIVE = "https://nsearchives.nseindia.com"
UDIFF_CUTOVER = date(2024, 7, 8)

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]


class RateLimiter:
    """Token-bucket-ish limiter shared across threads. Keeps us off NSE's radar."""

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / max(per_second, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            # tiny jitter so we don't look like a metronome
            time.sleep(random.uniform(0, 0.08))
            self._last = time.monotonic()


class NSEError(RuntimeError):
    pass


class NotAvailable(NSEError):
    """The file/endpoint exists but NSE has no data for that date (holiday, not yet published)."""


class NSEClient:
    def __init__(
        self,
        cache_dir: str | Path = "cache",
        requests_per_second: float = 2.5,
        timeout: int = 20,
        max_retries: int = 4,
        server_mode: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(requests_per_second)
        self.server_mode = server_mode
        self._ua = random.choice(_UA_POOL)
        self._session = None
        self._cookie_time = 0.0

    # ------------------------------------------------------------------ session

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self._ua,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
        }

    def _build_session(self):
        if self.server_mode:
            try:
                import httpx
            except ImportError as e:
                raise NSEError("server_mode=true needs `pip install httpx[http2]`") from e
            s = httpx.Client(http2=True, timeout=self.timeout, follow_redirects=True)
            s.headers.update(self.headers)
        else:
            import requests

            s = requests.Session()
            s.headers.update(self.headers)
        return s

    def _ensure_session(self, force: bool = False):
        """NSE cookies go stale in ~10 min of idling. Refresh generously."""
        stale = (time.time() - self._cookie_time) > 480
        if self._session is None or force or stale:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:
                    pass
            self._session = self._build_session()
            # Bootstrap: hit HTML pages so NSE sets nsit / nseappid / bm_sv cookies.
            for warm in ("https://www.nseindia.com/",
                         "https://www.nseindia.com/market-data/live-equity-market"):
                try:
                    self.limiter.wait()
                    self._session.get(warm, timeout=self.timeout)
                except Exception as e:  # non-fatal; archives often work anyway
                    log.debug("cookie warmup failed on %s: %s", warm, e)
            self._cookie_time = time.time()

    # ------------------------------------------------------------------- request

    def _raw_get(self, url: str, params: Optional[dict] = None, stream: bool = False):
        self._ensure_session()
        self.limiter.wait()
        if self.server_mode:
            return self._session.get(url, params=params)
        return self._session.get(url, params=params, timeout=self.timeout, stream=stream)

    def get(self, url: str, params: Optional[dict] = None) -> bytes:
        """GET with retry + session rebuild on 401/403 (the classic NSE cookie death)."""
        last = None
        for attempt in range(self.max_retries):
            try:
                r = self._raw_get(url, params=params)
                code = r.status_code
                if code in (401, 403, 429):
                    last = NSEError(f"{code} on {url}")
                    log.debug("NSE %s — rebuilding session (attempt %d)", code, attempt + 1)
                    self._ensure_session(force=True)
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if code == 404:
                    raise NotAvailable(f"404 {url}")
                if not 200 <= code < 300:
                    last = NSEError(f"{code} on {url}")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                content = r.content
                ctype = (r.headers.get("content-type") or "").lower()
                # NSE serves an HTML "file not available" page with a 200 status.
                if "text/html" in ctype and not url.endswith((".html", "/")):
                    raise NotAvailable(f"HTML placeholder returned for {url}")
                return content
            except NotAvailable:
                raise
            except Exception as e:  # network hiccup
                last = e
                log.debug("request error %s (attempt %d): %s", url, attempt + 1, e)
                self._ensure_session(force=True)
                time.sleep(1.2 * (attempt + 1))
        raise NSEError(f"failed after {self.max_retries} attempts: {url} ({last})")

    def get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = path if path.startswith("http") else f"{BASE}/{path.lstrip('/')}"
        raw = self.get(url, params=params)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise NSEError(f"non-JSON response from {url}: {raw[:120]!r}") from e

    # -------------------------------------------------------------------- files

    def _cached(self, key: str, max_age_hours: Optional[float] = None) -> Optional[bytes]:
        p = self.cache_dir / key
        if not p.exists():
            return None
        if max_age_hours is not None:
            age = (time.time() - p.stat().st_mtime) / 3600
            if age > max_age_hours:
                return None
        return p.read_bytes()

    def _store(self, key: str, data: bytes):
        p = self.cache_dir / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def fetch_file(self, url: str, cache_key: str,
                   max_age_hours: Optional[float] = None) -> bytes:
        """Download (or read from cache) a file, transparently unzipping/gunzipping."""
        hit = self._cached(cache_key, max_age_hours)
        if hit is not None:
            return hit
        raw = self.get(url)
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                name = z.namelist()[0]
                raw = z.read(name)
        elif url.endswith(".gz"):
            raw = gzip.decompress(raw)
        self._store(cache_key, raw)
        return raw

    # ---------------------------------------------------------------- bhavcopies

    def bhavcopy_full(self, d: date) -> bytes:
        """
        sec_bhavdata_full — OHLC + volume + turnover + DELIVERY for every equity.
        One request, whole market. Cached forever (historical files never change).
        """
        ds = d.strftime("%d%m%Y")
        url = f"{ARCHIVE}/products/content/sec_bhavdata_full_{ds}.csv"
        return self.fetch_file(url, f"bhav/full_{d:%Y%m%d}.csv")

    def bhavcopy_udiff(self, d: date) -> bytes:
        """UDiFF bhavcopy (or legacy format for pre-July-2024 dates)."""
        if d >= UDIFF_CUTOVER:
            url = f"{ARCHIVE}/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        else:
            mon = d.strftime("%b").upper()          # e.g. JAN
            fname = f"cm{d.strftime('%d%b%Y').upper()}bhav.csv.zip"
            url = f"{ARCHIVE}/content/historical/EQUITIES/{d.year}/{mon}/{fname}"
        return self.fetch_file(url, f"bhav/udiff_{d:%Y%m%d}.csv")

    def index_close_all(self, d: date) -> bytes:
        """Closing level of every NSE index for a date — benchmark & sector series."""
        url = f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv"
        return self.fetch_file(url, f"idx/close_{d:%Y%m%d}.csv")

    def security_list(self, d: date) -> bytes:
        """Security master with surveillance (ASM/GSM) flags."""
        url = f"{ARCHIVE}/content/equities/sec_list_{d:%d%m%Y}.csv"
        return self.fetch_file(url, f"sec/list_{d:%Y%m%d}.csv", max_age_hours=24)

    # -------------------------------------------------------------- F&O / flows

    def bhavcopy_fno(self, d: date) -> bytes:
        """
        UDiFF F&O bhavcopy — every futures and options contract for one day, with
        open interest and change in OI. Aggregated per underlying this gives
        futures OI build-up, the put-call ratio, and the futures basis.
        """
        url = f"{ARCHIVE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
        return self.fetch_file(url, f"fo/bhav_{d:%Y%m%d}.csv")

    def participant_oi(self, d: date) -> bytes:
        """
        Participant-wise open interest: how FIIs, DIIs, proprietary desks and
        retail clients are positioned in index and stock futures/options.
        NSE Clearing publishes this daily.
        """
        url = f"{ARCHIVE}/content/nsccl/fao_participant_oi_{d:%d%m%Y}.csv"
        return self.fetch_file(url, f"fo/participant_oi_{d:%Y%m%d}.csv")

    def participant_volume(self, d: date) -> bytes:
        url = f"{ARCHIVE}/content/nsccl/fao_participant_vol_{d:%d%m%Y}.csv"
        return self.fetch_file(url, f"fo/participant_vol_{d:%Y%m%d}.csv")

    def fii_dii_activity(self) -> Any:
        """
        FII and DII cash-market buy/sell for the latest session(s).
        This endpoint only returns the most recent days, so it must be polled
        daily and accumulated — there is no historical archive behind it.
        """
        return self.get_json("fiidiiTradeReact")

    # ------------------------------------------------------------------ JSON API

    def index_constituents(self, index: str = "NIFTY 500") -> List[dict]:
        endpoint = "equity-stock-indices"
        if index.upper() in ("PERMITTED TO TRADE", "SECURITIES IN F&O"):
            endpoint = "equity-stockIndex"
        data = self.get_json(endpoint, params={"index": index.upper()})
        return data.get("data", [])

    def all_indices(self) -> List[dict]:
        return self.get_json("allIndices").get("data", [])

    def quote(self, symbol: str) -> dict:
        return self.get_json("quote-equity", params={"symbol": symbol.upper()})

    def announcements(self, frm: date, to: date, index: str = "equities") -> List[dict]:
        params = {"index": index,
                  "from_date": frm.strftime("%d-%m-%Y"),
                  "to_date": to.strftime("%d-%m-%Y")}
        data = self.get_json("corporate-announcements", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def corporate_actions(self, frm: date, to: date) -> List[dict]:
        params = {"index": "equities",
                  "from_date": frm.strftime("%d-%m-%Y"),
                  "to_date": to.strftime("%d-%m-%Y")}
        data = self.get_json("corporates-corporateActions", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def financial_results(self, symbol: str, period: str = "Quarterly") -> List[dict]:
        params = {"index": "equities", "symbol": symbol.upper(),
                  "period": period, "fo_sec": "true"}
        data = self.get_json("corporates-financial-results", params=params)
        return data if isinstance(data, list) else data.get("data", [])

    def shareholding(self, symbol: str) -> Any:
        return self.get_json("corporate-share-holdings-master",
                             params={"symbol": symbol.upper(), "type": "equity"})

    def bulk_block_deals(self, frm: date, to: date) -> dict:
        params = {"index": "bulk_deals",
                  "from": frm.strftime("%d-%m-%Y"),
                  "to": to.strftime("%d-%m-%Y")}
        return self.get_json("historicalOR/bulk-block-short-deals", params=params)

    def market_status(self) -> List[dict]:
        return self.get_json("marketStatus").get("marketState", [])

    def holidays(self) -> dict:
        return self.get_json("holiday-master", params={"type": "trading"})

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False
