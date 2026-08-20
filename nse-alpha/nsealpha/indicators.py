"""
indicators.py — vectorised technical indicators.

All functions take pandas Series/DataFrames indexed by date for a *single*
symbol and return Series. They are deliberately dependency-free (no TA-Lib):
one less native build to fight with, and the formulas are short enough to audit.

Glossary of what each one is actually telling you:
  SMA / EMA      trend direction and dynamic support
  RSI            momentum oscillator, 0-100; >70 stretched, <30 washed out
  MACD           trend-following momentum; histogram = acceleration
  Bollinger      volatility envelope; %B = where price sits in the band
  ATR            average true range — the stock's normal daily travel, in rupees
  ADX / DI       trend *strength* (not direction); >25 = a real trend exists
  Supertrend     ATR-based trailing stop that flips long/short
  OBV            on-balance volume — is volume flowing in or out
  MFI            money flow index — RSI weighted by turnover
  CCI            commodity channel index — deviation from mean price
  Stochastic     where close sits within the recent high-low range
  Hurst          >0.5 trending, <0.5 mean-reverting, ~0.5 random walk
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------- averages

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


# ---------------------------------------------------------------- oscillators

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    # Wilder's smoothing
    ru = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def stochastic(high, low, close, n: int = 14, d: int = 3):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def cci(high, low, close, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def mfi(high, low, close, volume, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(), 0.0)
    neg = rmf.where(tp < tp.shift(), 0.0)
    pr = pos.rolling(n).sum()
    nr = neg.rolling(n).sum().replace(0, np.nan)
    return 100 - 100 / (1 + pr / nr)


# ----------------------------------------------------------------- volatility

def true_range(high, low, close) -> pd.Series:
    pc = close.shift()
    return pd.concat([(high - low).abs(),
                      (high - pc).abs(),
                      (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False,
                                            min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std()
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, width, pct_b


def keltner(high, low, close, n: int = 20, mult: float = 1.5):
    mid = ema(close, n)
    rng = atr(high, low, close, n) * mult
    return mid, mid + rng, mid - rng


# ------------------------------------------------------------- trend strength

def adx(high, low, close, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    mdi = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def supertrend(high, low, close, n: int = 10, mult: float = 3.0):
    """Returns (+1 uptrend / -1 downtrend) and the trailing stop line."""
    a = atr(high, low, close, n)
    hl2 = (high + low) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a

    fu = upper.copy()
    fl = lower.copy()
    u = upper.to_numpy(copy=True)
    l = lower.to_numpy(copy=True)
    c = close.to_numpy()
    for i in range(1, len(c)):
        if not np.isnan(u[i - 1]):
            u[i] = min(u[i], u[i - 1]) if c[i - 1] <= u[i - 1] else u[i]
            l[i] = max(l[i], l[i - 1]) if c[i - 1] >= l[i - 1] else l[i]
    fu[:] = u
    fl[:] = l

    dirn = np.ones(len(c))
    line = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        prev = dirn[i - 1]
        if prev == 1:
            dirn[i] = -1 if c[i] < l[i] else 1
        else:
            dirn[i] = 1 if c[i] > u[i] else -1
        line[i] = l[i] if dirn[i] == 1 else u[i]
    return pd.Series(dirn, index=close.index), pd.Series(line, index=close.index)


# --------------------------------------------------------------------- volume

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()


def vwap_rolling(high, low, close, volume, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    pv = (tp * volume).rolling(n).sum()
    v = volume.rolling(n).sum().replace(0, np.nan)
    return pv / v


def volume_surge(volume: pd.Series, n: int = 20) -> pd.Series:
    """Today's volume as a multiple of its own 20-day average."""
    return volume / volume.rolling(n, min_periods=5).mean().replace(0, np.nan)


# ----------------------------------------------------------- statistical shape

def slope(s: pd.Series, n: int) -> pd.Series:
    """Normalised linear-regression slope over n bars (per-bar % change)."""
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def f(y):
        if np.isnan(y).any():
            return np.nan
        b = ((x - xm) * (y - y.mean())).sum() / denom
        return b / (abs(y.mean()) + 1e-9)

    return s.rolling(n).apply(f, raw=True)


def hurst(s: pd.Series, max_lag: int = 40) -> float:
    """
    Rescaled-range style Hurst exponent via variance of lagged differences.
    >0.55 persistent/trending, <0.45 anti-persistent/mean-reverting.
    """
    x = np.asarray(s.dropna(), dtype=float)
    if len(x) < max_lag * 3:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        d = x[lag:] - x[:-lag]
        tau.append(np.sqrt(np.std(d)) if np.std(d) > 0 else np.nan)
    tau = np.asarray(tau, dtype=float)
    ok = ~np.isnan(tau) & (tau > 0)
    if ok.sum() < 5:
        return np.nan
    poly = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)
    return float(poly[0] * 2.0)


def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    peak = close.cummax()
    return float(((close / peak) - 1).min())


def realized_vol(close: pd.Series, n: int = 20, annualize: bool = True) -> pd.Series:
    r = np.log(close / close.shift())
    v = r.rolling(n, min_periods=max(3, n // 2)).std()
    return v * np.sqrt(252) * 100 if annualize else v


# -------------------------------------------------------------------- bundler

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: one symbol, columns [date(index), open, high, low, close, volume,
                             turnover, deliv_pct]
    Returns the same frame with every indicator column appended.
    """
    d = df.copy()
    h, l, c, v = d["high"], d["low"], d["close"], d["volume"]

    for n in (5, 10, 20, 50, 100, 200):
        d[f"sma{n}"] = sma(c, n)
    d["ema21"] = ema(c, 21)

    d["rsi14"] = rsi(c, 14)
    d["rsi_slope"] = d["rsi14"].diff(3)

    m, s, hist = macd(c)
    d["macd"], d["macd_signal"], d["macd_hist"] = m, s, hist
    d["macd_hist_slope"] = hist.diff(3)

    bb_mid, bb_up, bb_lo, bb_w, pct_b = bollinger(c)
    d["bb_upper"], d["bb_lower"] = bb_up, bb_lo
    d["bb_width"], d["bb_pct_b"] = bb_w, pct_b
    d["bb_squeeze"] = bb_w / bb_w.rolling(120, min_periods=40).median().replace(0, np.nan)

    d["atr14"] = atr(h, l, c, 14)
    d["atr_pct"] = 100 * d["atr14"] / c.replace(0, np.nan)

    a, pdi, mdi = adx(h, l, c, 14)
    d["adx"], d["plus_di"], d["minus_di"] = a, pdi, mdi

    st_dir, st_line = supertrend(h, l, c)
    d["supertrend_dir"], d["supertrend"] = st_dir, st_line

    d["obv"] = obv(c, v)
    d["obv_slope"] = slope(d["obv"], 20)
    d["vol_surge"] = volume_surge(v, 20)
    d["vwap20"] = vwap_rolling(h, l, c, v, 20)
    d["mfi14"] = mfi(h, l, c, v, 14)
    d["cci20"] = cci(h, l, c, 20)
    k, dd = stochastic(h, l, c)
    d["stoch_k"], d["stoch_d"] = k, dd

    d["hi_52w"] = c.rolling(252, min_periods=60).max()
    d["lo_52w"] = c.rolling(252, min_periods=60).min()
    d["pct_from_52w_high"] = 100 * (c / d["hi_52w"].replace(0, np.nan) - 1)
    d["pct_above_52w_low"] = 100 * (c / d["lo_52w"].replace(0, np.nan) - 1)
    d["hi_20d"] = h.rolling(20).max()

    d["vol20"] = realized_vol(c, 20)
    d["vol60"] = realized_vol(c, 60)

    for n, label in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        d[f"ret_{label}"] = 100 * (c / c.shift(n) - 1)
    d["ret_12m_1m"] = 100 * (c.shift(21) / c.shift(252) - 1)

    d["turnover_20d_cr"] = d["turnover"].rolling(20, min_periods=5).mean() / 1e7
    if "deliv_pct" in d:
        d["deliv_ma20"] = d["deliv_pct"].rolling(20, min_periods=5).mean()
        d["deliv_ma60"] = d["deliv_pct"].rolling(60, min_periods=15).mean()
        d["deliv_trend"] = d["deliv_ma20"] - d["deliv_ma60"]
    return d
