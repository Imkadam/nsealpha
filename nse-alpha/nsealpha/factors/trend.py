"""
trend.py — "past trends": what history says about *this* stock in *this* state.

The other four blocks describe the present. This one asks a different question:
when this particular stock has looked like this before, what happened next?

  historical_edge   For every past day we compute a lightweight setup score from
                    the same indicators the technical block uses. We then find
                    the historical days whose setup score sat in the same bucket
                    as today's, and average the forward 10-day return from those
                    days. This is a per-stock conditional expectation — a crude
                    but honest backtest running inside the factor itself. Stocks
                    whose current setup has historically preceded gains score
                    high; stocks where this same setup has repeatedly failed get
                    marked down no matter how pretty the chart looks today.
                    Look-ahead is avoided by only ever using data up to t and
                    forward returns that ended before today.

  regime_fit        Hurst exponent. A trend-following signal on a mean-reverting
                    stock is a losing trade dressed up as a system. H > 0.55 =
                    persistent (momentum works), H < 0.45 = anti-persistent
                    (fade the move). We reward alignment between the stock's
                    measured character and the direction our signals point.

  seasonality       Month-of-year effect for this stock over available history.
                    Real in Indian markets (results seasons, budget, monsoon-
                    linked sectors), but small — hence the modest weight.

  drawdown_health   Distance from the 52-week peak plus how the stock has
                    historically recovered from drawdowns of this size.

  volatility_regime Penalises a signal firing while volatility is expanding
                    (20d vol well above 60d vol) — that is usually distribution,
                    not accumulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators import hurst

FWD_DAYS = 10


def _setup_score_series(f: pd.DataFrame) -> pd.Series:
    """
    A compact, historically computable version of the technical block.
    Range roughly -6..+6. Used only for conditioning, never reported.
    """
    c = f["close"]
    s = (
        1.0 * (c > f["sma20"]).astype(float)
        + 1.0 * (c > f["sma50"]).astype(float)
        + 1.0 * (c > f["sma200"]).astype(float)
        + 1.0 * ((f["sma50"] > f["sma200"]).astype(float))
        + 1.0 * np.sign(f["macd_hist"].fillna(0))
        + 1.0 * f["supertrend_dir"].fillna(0)
        + 0.8 * np.clip((f["rsi14"] - 50) / 20, -1, 1)
        + 0.6 * np.clip((f["adx"] - 20) / 20, -1, 1.5)
        + 0.6 * np.clip(f["bb_pct_b"].fillna(0.5) * 2 - 1, -1, 1)
    )
    return s


def _historical_edge(f: pd.DataFrame) -> tuple[float, float, int]:
    """
    Returns (mean forward return %, hit rate, sample size) for historical days
    whose setup score was within +/-0.75 of today's.
    """
    if len(f) < 180:
        return np.nan, np.nan, 0
    s = _setup_score_series(f)
    c = f["close"]
    fwd = (c.shift(-FWD_DAYS) / c - 1) * 100

    # only rows where the forward window is fully in the past
    valid = fwd.notna() & s.notna()
    valid.iloc[-FWD_DAYS:] = False
    if valid.sum() < 60:
        return np.nan, np.nan, 0

    today = s.iloc[-1]
    if pd.isna(today):
        return np.nan, np.nan, 0

    band = 0.75
    sel = valid & (s.sub(today).abs() <= band)
    # widen the band until we have a usable sample
    while sel.sum() < 25 and band < 3.0:
        band += 0.5
        sel = valid & (s.sub(today).abs() <= band)
    if sel.sum() < 15:
        return np.nan, np.nan, int(sel.sum())

    sample = fwd[sel]
    return float(sample.mean()), float((sample > 0).mean()), int(len(sample))


def _seasonality(f: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Mean forward-21-day return of this stock, historically, in this month."""
    if len(f) < 400 or not isinstance(f.index, pd.DatetimeIndex):
        return np.nan
    c = f["close"]
    fwd = (c.shift(-21) / c - 1) * 100
    month = pd.Series(f.index.month, index=f.index)
    cur = as_of.month
    sel = (month == cur) & fwd.notna()
    if sel.sum() < 15:
        return np.nan
    overall = fwd.dropna().mean()
    return float(fwd[sel].mean() - overall)


def _drawdown_health(f: pd.DataFrame) -> float:
    c = f["close"]
    if len(c) < 120:
        return np.nan
    peak = c.rolling(252, min_periods=60).max()
    dd_now = float(c.iloc[-1] / peak.iloc[-1] - 1) * 100
    dd = (c / peak - 1) * 100
    # how often has this stock recovered to a new high within 60 days of a
    # drawdown of similar depth?
    similar = dd.dropna()
    similar = similar[(similar - dd_now).abs() < 4]
    if len(similar) < 20:
        return dd_now / 10
    idxs = [c.index.get_loc(i) for i in similar.index if c.index.get_loc(i) + 60 < len(c)]
    if not idxs:
        return dd_now / 10
    rec = np.mean([1.0 if c.iloc[i:i + 60].max() >= peak.iloc[i] else 0.0 for i in idxs])
    return float(dd_now / 10 + rec * 3)


def compute(ctx) -> pd.DataFrame:
    idx = ctx.snapshot.index
    as_of = ctx.as_of if ctx.as_of is not None else pd.Timestamp.today()
    rows = {}

    for sym in idx:
        f = ctx.frames.get(sym)
        if f is None or f.empty:
            rows[sym] = dict(historical_edge=np.nan, regime_fit=np.nan,
                             seasonality=np.nan, drawdown_health=np.nan,
                             volatility_regime=np.nan, _edge_ret=np.nan,
                             _edge_hit=np.nan, _edge_n=0, _hurst=np.nan)
            continue

        mean_fwd, hit, n = _historical_edge(f)
        # blend size of edge with reliability of edge, discount tiny samples
        conf = np.clip(n / 60, 0, 1)
        edge = np.nan
        if not np.isnan(mean_fwd):
            edge = (np.clip(mean_fwd, -8, 8) * 0.6 +
                    (hit - 0.5) * 12 * 0.4) * (0.5 + 0.5 * conf)

        h = hurst(np.log(f["close"].dropna()))
        setup_dir = np.sign(_setup_score_series(f).iloc[-1]) if len(f) else 0
        if np.isnan(h):
            regime = np.nan
        else:
            persistence = (h - 0.5) * 8          # -0.4..+0.4 -> -3.2..+3.2
            # trend-following signals deserve credit only on persistent names
            regime = persistence * (1 if setup_dir >= 0 else -1)

        seas = _seasonality(f, as_of)
        ddh = _drawdown_health(f)

        v20 = f["vol20"].iloc[-1] if "vol20" in f else np.nan
        v60 = f["vol60"].iloc[-1] if "vol60" in f else np.nan
        if pd.isna(v20) or pd.isna(v60) or v60 == 0:
            volreg = np.nan
        else:
            ratio = v20 / v60
            # 0.8-1.1 is calm/steady = good; >1.5 = vol explosion = bad
            volreg = float(np.clip((1.25 - ratio) * 4, -3, 3))

        rows[sym] = dict(
            historical_edge=edge,
            regime_fit=regime,
            seasonality=np.nan if np.isnan(seas) else float(np.clip(seas, -6, 6)),
            drawdown_health=ddh,
            volatility_regime=volreg,
            _edge_ret=mean_fwd, _edge_hit=hit, _edge_n=n, _hurst=h,
        )

    return pd.DataFrame.from_dict(rows, orient="index").reindex(idx)
