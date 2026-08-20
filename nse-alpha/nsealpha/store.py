"""
store.py — SQLite persistence layer.

Everything the model needs lives in one file on disk. Rebuilding history from
NSE archives takes a while (one request per trading day), so we do it once and
then only append the newest day. Re-running the pipeline on a Tuesday evening
should cost exactly one bhavcopy download.

Tables
------
prices        : symbol, date, OHLC, volume, turnover, trades, delivery qty/pct, series
indices       : index_name, date, close, pct_change  (benchmark + sector series)
fundamentals  : symbol, as_of, JSON blob of ratios (refreshed weekly)
announcements : symbol, date, subject, description, sentiment score
deals         : symbol, date, client, buy/sell, qty, price (bulk & block)
meta          : key/value scratchpad (last ingest date, universe snapshot, ...)
scores        : the full daily ranking, so we can audit what we said and when
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol        TEXT NOT NULL,
    date          TEXT NOT NULL,
    series        TEXT,
    open          REAL, high REAL, low REAL, close REAL, prev_close REAL,
    last_price    REAL,
    volume        REAL,          -- shares traded
    turnover      REAL,          -- rupees
    trades        REAL,
    deliv_qty     REAL,
    deliv_pct     REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS indices (
    index_name TEXT NOT NULL,
    date       TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    pct_change REAL,
    PRIMARY KEY (index_name, date)
);

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT NOT NULL,
    as_of  TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, as_of)
);

CREATE TABLE IF NOT EXISTS announcements (
    symbol TEXT, date TEXT, subject TEXT, detail TEXT,
    score REAL, tags TEXT,
    PRIMARY KEY (symbol, date, subject)
);

CREATE TABLE IF NOT EXISTS deals (
    symbol TEXT, date TEXT, client TEXT, side TEXT,
    qty REAL, price REAL, kind TEXT
);
CREATE INDEX IF NOT EXISTS ix_deals ON deals(symbol, date);

CREATE TABLE IF NOT EXISTS security_master (
    symbol TEXT PRIMARY KEY,
    name TEXT, isin TEXT, series TEXT,
    listing_date TEXT, face_value REAL,
    surveillance TEXT, sector TEXT, industry TEXT,
    updated TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    run_date TEXT, symbol TEXT, rank INTEGER,
    composite REAL, technical REAL, momentum REAL,
    fundamental REAL, sentiment REAL, trend REAL,
    detail TEXT,
    PRIMARY KEY (run_date, symbol)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol TEXT, ex_date TEXT, purpose TEXT, record_date TEXT,
    PRIMARY KEY (symbol, ex_date, purpose)
);

-- Per-underlying F&O positioning, derived from the daily FO bhavcopy.
CREATE TABLE IF NOT EXISTS fno_daily (
    symbol TEXT, date TEXT,
    fut_close REAL, fut_oi REAL, fut_oi_chg REAL, fut_volume REAL,
    call_oi REAL, put_oi REAL, call_volume REAL, put_volume REAL,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS ix_fno_date ON fno_daily(date);

-- Market-wide institutional activity (one row per day, not per stock).
CREATE TABLE IF NOT EXISTS market_flows (
    date TEXT NOT NULL, metric TEXT NOT NULL, value REAL,
    PRIMARY KEY (date, metric)
);

-- Quarterly shareholding pattern per company.
CREATE TABLE IF NOT EXISTS shareholding (
    symbol TEXT, period TEXT,
    promoter REAL, pledge REAL, fii REAL, dii REAL, public REAL,
    PRIMARY KEY (symbol, period)
);

-- Annual income statement / balance sheet / cash flow, in long format so new
-- line items never need a schema migration.
CREATE TABLE IF NOT EXISTS financial_statements (
    symbol TEXT NOT NULL, period TEXT NOT NULL, item TEXT NOT NULL,
    value REAL, fetched TEXT,
    PRIMARY KEY (symbol, period, item)
);
CREATE INDEX IF NOT EXISTS ix_stmt_symbol ON financial_statements(symbol);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is required because the Streamlit app caches one
        # Store and Streamlit runs each browser session on its own thread. A raw
        # sqlite3 connection is not safe for *concurrent* use though, so every
        # public method is serialised on the lock below (see the wrap at the
        # bottom of this file). Without both halves you get either
        # "SQLite objects created in a thread can only be used in that same
        # thread" or, worse, interleaved statements and silent corruption.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ helpers

    @contextmanager
    def tx(self):
        # The lock is taken inside the generator body so it is held for the whole
        # `with` block, not merely while the context manager is constructed.
        with self._lock:
            try:
                yield self.conn
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def set_meta(self, key: str, value):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                      (key, json.dumps(value, default=str)))

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ------------------------------------------------------------------- prices

    def upsert_prices(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "date", "series", "open", "high", "low", "close",
                "prev_close", "last_price", "volume", "turnover", "trades",
                "deliv_qty", "deliv_pct"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        with self.tx() as c:
            c.executemany(
                f"INSERT OR REPLACE INTO prices ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                df.itertuples(index=False, name=None),
            )
        return len(df)

    def price_dates(self) -> set:
        rows = self.conn.execute("SELECT DISTINCT date FROM prices").fetchall()
        return {r[0] for r in rows}

    def load_prices(self, since: Optional[str] = None,
                    symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        q = "SELECT * FROM prices"
        clauses, params = [], []
        if since:
            clauses.append("date >= ?")
            params.append(since)
        if symbols:
            symbols = list(symbols)
            clauses.append(f"symbol IN ({','.join('?' * len(symbols))})")
            params.extend(symbols)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY symbol, date"
        df = pd.read_sql_query(q, self.conn, params=params, parse_dates=["date"])
        return df

    def latest_price_date(self) -> Optional[str]:
        row = self.conn.execute("SELECT MAX(date) FROM prices").fetchone()
        return row[0] if row and row[0] else None

    # ------------------------------------------------------------------ indices

    def upsert_indices(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["index_name", "date", "open", "high", "low", "close", "pct_change"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        with self.tx() as c:
            c.executemany(
                f"INSERT OR REPLACE INTO indices ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                df.itertuples(index=False, name=None),
            )
        return len(df)

    def load_index(self, name: str, since: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT date, close FROM indices WHERE UPPER(index_name)=?"
        params = [name.upper()]
        if since:
            q += " AND date >= ?"
            params.append(since)
        q += " ORDER BY date"
        return pd.read_sql_query(q, self.conn, params=params, parse_dates=["date"])

    def index_names(self) -> list:
        return [r[0] for r in
                self.conn.execute("SELECT DISTINCT index_name FROM indices").fetchall()]

    # ------------------------------------------------------------- fundamentals

    def upsert_fundamentals(self, symbol: str, as_of: str, payload: dict):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO fundamentals VALUES (?,?,?)",
                      (symbol, as_of, json.dumps(payload, default=str)))

    def load_fundamentals(self) -> pd.DataFrame:
        q = """SELECT f.symbol, f.as_of, f.payload FROM fundamentals f
               JOIN (SELECT symbol, MAX(as_of) m FROM fundamentals GROUP BY symbol) x
                 ON f.symbol = x.symbol AND f.as_of = x.m"""
        rows = self.conn.execute(q).fetchall()
        if not rows:
            return pd.DataFrame()
        recs = []
        for sym, as_of, payload in rows:
            d = json.loads(payload)
            d["symbol"] = sym
            d["as_of"] = as_of
            recs.append(d)
        return pd.DataFrame(recs)

    def fundamentals_age_days(self, symbol: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT MAX(as_of) FROM fundamentals WHERE symbol=?", (symbol,)).fetchone()
        if not row or not row[0]:
            return None
        return (date.today() - datetime.strptime(row[0], "%Y-%m-%d").date()).days

    # ------------------------------------------------------------ announcements

    def upsert_announcements(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "date", "subject", "detail", "score", "tags"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        with self.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO announcements VALUES (?,?,?,?,?,?)",
                df[cols].itertuples(index=False, name=None),
            )
        return len(df)

    def load_announcements(self, since: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM announcements WHERE date >= ? ORDER BY date",
            self.conn, params=[since], parse_dates=["date"])

    # -------------------------------------------------------------------- deals

    def upsert_deals(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "date", "client", "side", "qty", "price", "kind"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        with self.tx() as c:
            c.execute("DELETE FROM deals WHERE date IN ({})".format(
                ",".join("?" * df["date"].nunique())), list(df["date"].unique()))
            c.executemany("INSERT INTO deals VALUES (?,?,?,?,?,?,?)",
                          df[cols].itertuples(index=False, name=None))
        return len(df)

    def load_deals(self, since: str) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM deals WHERE date >= ?",
                                 self.conn, params=[since], parse_dates=["date"])

    # ---------------------------------------------------- financial statements

    def upsert_statements(self, symbol: str, wide: pd.DataFrame, fetched: str):
        """`wide` is period-indexed, one column per line item."""
        if wide is None or wide.empty:
            return 0
        rows = []
        for period, row in wide.iterrows():
            p = pd.Timestamp(period).strftime("%Y-%m-%d")
            for item, value in row.items():
                v = pd.to_numeric(value, errors="coerce")
                if pd.isna(v):
                    continue
                rows.append((symbol, p, str(item), float(v), fetched))
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO financial_statements VALUES (?,?,?,?,?)", rows)
        return len(rows)

    def load_statements(self, symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        q, params = "SELECT symbol, period, item, value FROM financial_statements", []
        if symbols:
            symbols = list(symbols)
            q += f" WHERE symbol IN ({','.join('?' * len(symbols))})"
            params.extend(symbols)
        return pd.read_sql_query(q, self.conn, params=params)

    def statements_age_days(self, symbol: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT MAX(fetched) FROM financial_statements WHERE symbol=?",
            (symbol,)).fetchone()
        if not row or not row[0]:
            return None
        try:
            return (date.today() - datetime.strptime(row[0], "%Y-%m-%d").date()).days
        except Exception:
            return None

    # -------------------------------------------------------- corporate actions

    def upsert_corporate_actions(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "ex_date", "purpose", "record_date"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        with self.tx() as c:
            c.executemany("INSERT OR REPLACE INTO corporate_actions VALUES (?,?,?,?)",
                          df[cols].itertuples(index=False, name=None))
        return len(df)

    def load_corporate_actions(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM corporate_actions", self.conn)

    # --------------------------------------------------------------- F&O flows

    def upsert_fno(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "date", "fut_close", "fut_oi", "fut_oi_chg", "fut_volume",
                "call_oi", "put_oi", "call_volume", "put_volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        d = df[cols].copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        with self.tx() as c:
            c.executemany(
                f"INSERT OR REPLACE INTO fno_daily ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                d.itertuples(index=False, name=None))
        return len(d)

    def load_fno(self, since: Optional[str] = None) -> pd.DataFrame:
        q, params = "SELECT * FROM fno_daily", []
        if since:
            q += " WHERE date >= ?"
            params.append(since)
        q += " ORDER BY symbol, date"
        return pd.read_sql_query(q, self.conn, params=params, parse_dates=["date"])

    def fno_dates(self) -> set:
        return {r[0] for r in
                self.conn.execute("SELECT DISTINCT date FROM fno_daily").fetchall()}

    # ------------------------------------------------------- market-wide flows

    def upsert_market_flows(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        d = df[["date", "metric", "value"]].copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
        with self.tx() as c:
            c.executemany("INSERT OR REPLACE INTO market_flows VALUES (?,?,?)",
                          d.itertuples(index=False, name=None))
        return len(d)

    def load_market_flows(self, since: Optional[str] = None) -> pd.DataFrame:
        q, params = "SELECT * FROM market_flows", []
        if since:
            q += " WHERE date >= ?"
            params.append(since)
        return pd.read_sql_query(q + " ORDER BY date", self.conn, params=params,
                                 parse_dates=["date"])

    # --------------------------------------------------------- shareholding

    def upsert_shareholding(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "period", "promoter", "pledge", "fii", "dii", "public"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        with self.tx() as c:
            c.executemany("INSERT OR REPLACE INTO shareholding VALUES (?,?,?,?,?,?,?)",
                          df[cols].itertuples(index=False, name=None))
        return len(df)

    def load_shareholding(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM shareholding ORDER BY symbol, period",
                                 self.conn)

    # ----------------------------------------------------------- security master

    def upsert_securities(self, df: pd.DataFrame):
        if df is None or df.empty:
            return 0
        cols = ["symbol", "name", "isin", "series", "listing_date", "face_value",
                "surveillance", "sector", "industry", "updated"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        with self.tx() as c:
            c.executemany(
                f"INSERT OR REPLACE INTO security_master ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                df[cols].itertuples(index=False, name=None))
        return len(df)

    def load_securities(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM security_master", self.conn)

    # ------------------------------------------------------------------- scores

    def save_scores(self, run_date: str, df: pd.DataFrame):
        with self.tx() as c:
            c.execute("DELETE FROM scores WHERE run_date=?", (run_date,))
            rows = []
            for r in df.itertuples(index=False):
                d = r._asdict()
                rows.append((
                    run_date, d["symbol"], int(d.get("rank", 0)),
                    float(d.get("composite", 0) or 0),
                    float(d.get("technical", 0) or 0),
                    float(d.get("momentum", 0) or 0),
                    float(d.get("fundamental", 0) or 0),
                    float(d.get("sentiment", 0) or 0),
                    float(d.get("trend", 0) or 0),
                    json.dumps({k: (None if pd.isna(v) else v)
                                for k, v in d.items()
                                if isinstance(v, (int, float, str, type(None)))},
                               default=str),
                ))
            c.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    def load_scores(self, run_date: str) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM scores WHERE run_date=? ORDER BY rank",
                                 self.conn, params=[run_date])

    def score_history(self, limit: int = 60) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM scores WHERE run_date IN "
            "(SELECT DISTINCT run_date FROM scores ORDER BY run_date DESC LIMIT ?) "
            "ORDER BY run_date DESC, rank", self.conn, params=[limit])

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------------------
# Serialise every public method on the Store's own lock.
#
# Done here in one place rather than by decorating twenty-five methods by hand:
# it is impossible to add a new query later and forget the lock, which is exactly
# the kind of bug that only shows up under a second concurrent user and then
# corrupts data rather than raising. The lock is re-entrant, so a write method
# calling tx() internally nests safely.
# ---------------------------------------------------------------------------

def _synchronized(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


for _name, _member in list(vars(Store).items()):
    if _name.startswith("_") or _name == "tx" or not callable(_member):
        continue
    setattr(Store, _name, _synchronized(_member))
