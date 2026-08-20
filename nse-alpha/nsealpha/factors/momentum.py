"""
momentum.py — the "winners keep winning" block, plus relative strength.

Cross-sectional momentum is the most durably documented equity anomaly there is,
and it works on Indian equities too. Three deliberate design choices:

  1. ret_12m_1m skips the most recent month. Twelve-month momentum minus the last
     month is the academic standard because the very last month tends to
     short-term *reverse*. Including it dilutes the signal.
  2. Relative strength is measured against the benchmark, not in absolute terms
     (Mansfield RS). In a bull market everything is up; what matters is what is
     up *more than the index*, and whether that gap is still widening.
  3. Consistency matters as much as size. A stock that ground out +40% over six
     months on low volatility is a better hold than one that gapped +40% in a
     week. We measure that with a return/volatility ratio and the share of
     positive weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rel_strength(ctx) -> pd.DataFrame:
    """Mansfield RS: stock/benchmark ratio vs its own 52-week average, plus slope."""
    bench = ctx.benchmark.set_index("date")["close"].sort_index()
    rows = {}
    for sym, f in ctx.frames.items():
        c = f["close"]
        b = bench.reindex(c.index).ffill()
        if b.isna().all() or len(c) < 60:
            rows[sym] = (np.nan, np.nan, np.nan)
            continue
        rs = c / b.replace(0, np.nan)
        base = rs.rolling(252, min_periods=60).mean()
        mansfield = 100 * (rs / base.replace(0, np.nan) - 1)
        # is the RS line itself still rising?
        rs_slope_13w = rs.iloc[-1] / rs.iloc[-65] - 1 if len(rs) > 65 else np.nan
        rs_slope_4w = rs.iloc[-1] / rs.iloc[-21] - 1 if len(rs) > 21 else np.nan
        rows[sym] = (float(mansfield.iloc[-1]) if pd.notna(mansfield.iloc[-1]) else np.nan,
                     float(rs_slope_13w) if pd.notna(rs_slope_13w) else np.nan,
                     float(rs_slope_4w) if pd.notna(rs_slope_4w) else np.nan)
    return pd.DataFrame(rows, index=["mansfield_rs", "rs_13w", "rs_4w"]).T


def _consistency(ctx) -> pd.DataFrame:
    rows = {}
    for sym, f in ctx.frames.items():
        c = f["close"].tail(130)
        if len(c) < 60:
            rows[sym] = (np.nan, np.nan, np.nan)
            continue
        wk = c.resample("W").last().dropna() if isinstance(c.index, pd.DatetimeIndex) \
            else c.iloc[::5]
        wr = wk.pct_change().dropna()
        pos_share = float((wr > 0).mean()) if len(wr) else np.nan
        d = c.pct_change().dropna()
        sharpe = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else np.nan
        # largest single-day contribution: high = the whole move was one gap
        total = float(c.iloc[-1] / c.iloc[0] - 1)
        concentration = float(d.abs().max() / (abs(total) + 1e-6)) if total else np.nan
        rows[sym] = (pos_share, sharpe, concentration)
    return pd.DataFrame(rows, index=["pos_weeks", "sharpe", "concentration"]).T


def compute(ctx) -> pd.DataFrame:
    s = ctx.snapshot
    out = pd.DataFrame(index=s.index)

    # Raw price momentum over several horizons. Clipped so one 300% moonshot
    # doesn't dominate the cross-sectional z-score before winsorisation.
    out["ret_1m"] = np.clip(s["ret_1m"], -60, 60)
    out["ret_3m"] = np.clip(s["ret_3m"], -80, 150)
    out["ret_6m"] = np.clip(s["ret_6m"], -90, 250)
    out["ret_12m_1m"] = np.clip(s["ret_12m_1m"], -90, 400)

    rs = _rel_strength(ctx).reindex(s.index)
    cons = _consistency(ctx).reindex(s.index)

    # Relative strength: level of the Mansfield line + whether it is still rising.
    out["rs_vs_market"] = (
        1.0 * np.clip(rs["mansfield_rs"].fillna(0) / 10, -3, 3) +
        1.0 * np.clip(rs["rs_13w"].fillna(0) * 10, -3, 3) +
        0.6 * np.clip(rs["rs_4w"].fillna(0) * 12, -3, 3)
    )

    # Consistency: smooth grinders beat one-gap wonders.
    out["consistency"] = (
        2.0 * (cons["pos_weeks"].fillna(0.5) - 0.5) * 4 +
        1.5 * np.clip(cons["sharpe"].fillna(0), -2.5, 3.5) -
        1.0 * np.clip(cons["concentration"].fillna(0.2), 0, 1.5)
    )

    # keep the diagnostics for the report
    out["_mansfield_rs"] = rs["mansfield_rs"]
    out["_pos_weeks"] = cons["pos_weeks"]
    out["_sharpe"] = cons["sharpe"]
    return out
