"""
technical.py — chart-based evidence that a stock is in a healthy uptrend *now*.

Sub-factors (all oriented higher = better):

  trend_structure     Stage analysis. Full marks when close > SMA20 > SMA50 >
                      SMA200 and the 200-DMA itself is rising ("Stage 2 advance"
                      in Weinstein's language). A golden cross adds; a death
                      cross subtracts.
  adx_strength        ADX(14) says how strong the trend is, +DI vs -DI says which
                      way. ADX 20-40 with +DI on top is the sweet spot. ADX above
                      ~50 usually means the move is late.
  rsi_zone            Deliberately non-monotonic. RSI 55-70 = strong but not
                      exhausted, scores best. RSI > 80 = overbought, penalised.
                      RSI < 40 in a downtrend = falling knife, penalised.
  macd                Histogram positive AND expanding beats histogram positive
                      and shrinking. A fresh bullish crossover is a bonus.
  bollinger           Rewards a stock breaking out of a volatility squeeze
                      (bb_width compressed vs its own history, %B pushing above
                      0.8) rather than one already glued to the upper band.
  supertrend          +1/-1 direction plus how far price sits above the trailing
                      stop, in ATR units (buffer before the trend flips).
  volume_confirmation OBV rising with price + today's volume vs its 20d average.
                      Price without volume is a rumour.
  breakout_proximity  Near the 52-week high is bullish on NSE as elsewhere;
                      an actual close above the 20-day high is the trigger.
  delivery_quality    NSE-specific. Rising delivery % alongside rising price
                      means real buyers taking shares home, not intraday churn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _z(x):
    return x


def _clip01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


def compute(ctx) -> pd.DataFrame:
    s = ctx.snapshot
    out = pd.DataFrame(index=s.index)
    c = s["close"]

    # ---------------------------------------------------------- trend structure
    above20 = (c > s["sma20"]).astype(float)
    above50 = (c > s["sma50"]).astype(float)
    above200 = (c > s["sma200"]).astype(float)
    stacked = ((s["sma20"] > s["sma50"]) & (s["sma50"] > s["sma200"])).astype(float)
    sma200_rising = (s["sma200"] > s["sma200_prev20"]).astype(float)
    golden = ((s["sma50"] > s["sma200"]) & (s["sma50_prev20"] <= s["sma200_prev20"])).astype(float)
    death = ((s["sma50"] < s["sma200"]) & (s["sma50_prev20"] >= s["sma200_prev20"])).astype(float)
    # how far above the 50-DMA, in ATR units — extension, not just position
    ext = (c - s["sma50"]) / s["atr14"].replace(0, np.nan)
    ext_score = np.where(ext > 6, 0.3, _clip01(ext, -1.5, 3.0))   # too extended = risky

    out["trend_structure"] = (
        1.2 * above20 + 1.5 * above50 + 2.0 * above200 +
        2.0 * stacked + 1.5 * sma200_rising +
        1.0 * golden - 2.0 * death +
        1.5 * pd.Series(ext_score, index=s.index)
    )

    # ------------------------------------------------------------ ADX strength
    adx = s["adx"]
    di_edge = (s["plus_di"] - s["minus_di"]) / (s["plus_di"] + s["minus_di"]).replace(0, np.nan)
    adx_shape = np.where(adx < 15, 0.2,                       # no trend at all
                 np.where(adx <= 40, _clip01(adx, 15, 32),    # healthy trend
                          np.clip(1.0 - (adx - 40) / 40, 0.3, 1.0)))  # exhausted
    out["adx_strength"] = pd.Series(adx_shape, index=s.index) * (0.5 + 0.5 * di_edge.fillna(0)) * 4

    # ----------------------------------------------------------------- RSI zone
    r = s["rsi14"]
    rsi_shape = np.select(
        [r >= 82, (r >= 70) & (r < 82), (r >= 55) & (r < 70),
         (r >= 45) & (r < 55), (r >= 35) & (r < 45), r < 35],
        [0.15, 0.65, 1.00, 0.70, 0.35, 0.10],
        default=0.5)
    # rising RSI is worth more than falling RSI at the same level
    out["rsi_zone"] = pd.Series(rsi_shape, index=s.index) * 3 + \
        np.clip(s["rsi_slope"].fillna(0) / 10, -1, 1)

    # --------------------------------------------------------------------- MACD
    hist_norm = s["macd_hist"] / c.replace(0, np.nan) * 100
    out["macd"] = (
        2.0 * np.sign(s["macd_hist"].fillna(0)) +
        1.5 * np.clip(hist_norm.fillna(0) * 4, -2, 2) +
        1.5 * np.sign(s["macd_hist_slope"].fillna(0)) +
        1.0 * ((s["macd"] > s["macd_signal"]) & (s["macd_prev5"] <= s["macd_signal_prev5"])).astype(float)
    )

    # ---------------------------------------------------------------- Bollinger
    pb = s["bb_pct_b"]
    pb_shape = np.select(
        [pb > 1.05, (pb > 0.80) & (pb <= 1.05), (pb > 0.55) & (pb <= 0.80),
         (pb > 0.25) & (pb <= 0.55), pb <= 0.25],
        [0.55, 1.00, 0.80, 0.45, 0.15], default=0.5)
    squeeze = np.clip(1.4 - s["bb_squeeze"].fillna(1.0), 0, 1)   # width vs own median
    out["bollinger"] = pd.Series(pb_shape, index=s.index) * 3 + squeeze * 1.5

    # --------------------------------------------------------------- Supertrend
    buffer_atr = (c - s["supertrend"]) / s["atr14"].replace(0, np.nan)
    out["supertrend"] = (2.5 * s["supertrend_dir"].fillna(0) +
                         1.5 * np.clip(buffer_atr.fillna(0), -3, 3))

    # ------------------------------------------------------ volume confirmation
    obv_up = np.sign(s["obv_slope"].fillna(0))
    price_up = np.sign(s["ret_1m"].fillna(0))
    agree = (obv_up == price_up).astype(float) * obv_up      # +1 confirms up, -1 confirms down
    surge = np.clip(s["vol_surge"].fillna(1.0), 0.2, 4.0)
    surge_shape = np.where(surge < 0.7, -0.5, np.clip((surge - 1.0) * 1.2, -0.5, 2.0))
    mfi_shape = _clip01(s["mfi14"].fillna(50), 35, 75)
    out["volume_confirmation"] = (2.0 * agree +
                                  1.5 * pd.Series(surge_shape, index=s.index) +
                                  1.5 * pd.Series(mfi_shape, index=s.index))

    # ---------------------------------------------------- breakout / proximity
    near_high = 1 + s["pct_from_52w_high"].fillna(-50) / 25      # 0% away -> 1.0, -25% -> 0
    broke_20d = (c >= s["hi_20d"] * 0.999).astype(float)
    off_low = _clip01(s["pct_above_52w_low"].fillna(0), 10, 120)
    out["breakout_proximity"] = (2.0 * np.clip(near_high, -1, 1) +
                                 1.5 * broke_20d +
                                 1.0 * pd.Series(off_low, index=s.index))

    # ---------------------------------------------------------- delivery quality
    if "deliv_ma20" in s.columns:
        lvl = _clip01(s["deliv_ma20"].fillna(np.nan), 25, 65)
        trend = np.clip(s["deliv_trend"].fillna(0) / 8, -1.5, 1.5)
        # accumulation = delivery improving while price is up
        accum = np.sign(s["deliv_trend"].fillna(0)) * np.sign(s["ret_1m"].fillna(0))
        out["delivery_quality"] = (2.0 * pd.Series(lvl, index=s.index).fillna(0.5) +
                                   1.5 * trend + 1.0 * accum)
    else:
        out["delivery_quality"] = np.nan

    return out
