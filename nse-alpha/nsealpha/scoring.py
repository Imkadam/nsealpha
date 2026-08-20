"""
scoring.py — turn raw sub-factors into one comparable number per stock.

The statistics, kept in one place so they can be argued with:

  winsorise    Clip each factor at its 2nd/98th percentile before standardising.
               One stock that quintupled would otherwise define the entire
               momentum scale and squash everyone else to zero.
  z-score      Standardise cross-sectionally: (x - mean) / std, computed across
               today's universe. This makes "good RSI" and "good ROE"
               commensurable without hand-tuned magic numbers.
  sector-neutral
               Optionally z-score *within* each sector. Without it, a screen run
               in a month when PSU banks are ripping returns ten PSU banks — you
               have bought one bet ten times. With it, you get the best name in
               each sector relative to its peers.
  coverage     A stock must have real values for at least `min_factor_coverage`
               of the sub-factors, otherwise it is dropped rather than ranked on
               fragments. Missing sub-factors inside a block are renormalised
               away, not treated as zero.
  clip         Final block z-scores are clipped to +/-3 so no single block can
               run away with the composite.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger("nsealpha.scoring")

BLOCKS = ["technical", "momentum", "fundamental", "sentiment", "trend", "flows"]


def winsorize(s: pd.Series, pct: float) -> pd.Series:
    if s.notna().sum() < 10 or pct <= 0:
        return s
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index).where(s.notna())
    return (s - m) / sd


def zscore_by_group(s: pd.Series, groups: pd.Series, min_group: int = 8) -> pd.Series:
    """
    Sector-neutral z-score, falling back to the global z-score for thin sectors.

    Uses groupby.transform("mean"/"std") rather than transform(python_function).
    Both give the same answer, but the named aggregations take pandas' Cython path
    while a Python callable is applied group-by-group in the interpreter. Across
    ~100 sub-factor columns that difference is most of the app's re-score time,
    which is what stands between the weight sliders feeling live and feeling laggy.
    """
    g = groups.reindex(s.index).fillna("UNKNOWN")
    counts = g.value_counts()
    big = set(counts[counts >= min_group].index)
    out = pd.Series(np.nan, index=s.index, dtype=float)

    mask_big = g.isin(big)
    if mask_big.any():
        sb, gb = s[mask_big], g[mask_big]
        grp = sb.groupby(gb, sort=False)
        mean = grp.transform("mean")
        std = grp.transform("std", ddof=0)
        z = (sb - mean) / std.where(np.isfinite(std) & (std != 0))
        # A sector where every stock has the identical value carries no
        # cross-sectional information: score it flat rather than NaN, so those
        # stocks aren't silently dropped for lack of coverage.
        out[mask_big] = z.where(z.notna(), other=sb.notna().map({True: 0.0,
                                                                False: np.nan}))
    if (~mask_big).any():
        out[~mask_big] = zscore(s[~mask_big]) if (~mask_big).sum() > 3 else 0.0
    return out


def standardise(df: pd.DataFrame, cfg: dict, sectors: pd.Series | None = None) -> pd.DataFrame:
    """Winsorise then z-score every column of a raw sub-factor frame."""
    pct = cfg["scoring"].get("winsorize_pct", 0.02)
    neutral = cfg["scoring"].get("sector_neutral", False) and sectors is not None
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c.startswith("_"):              # diagnostics pass through untouched
            out[c] = df[c]
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 5:
            out[c] = np.nan
            continue
        s = winsorize(s, pct)
        out[c] = zscore_by_group(s, sectors) if neutral else zscore(s)
    return out


def blend(df: pd.DataFrame, weights: Dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """
    Weighted mean of the named columns, ignoring NaNs by renormalising the
    weights that are actually available for each row.
    Returns (blended score, coverage fraction).
    """
    cols = [c for c in weights if c in df.columns]
    if not cols:
        return (pd.Series(np.nan, index=df.index),
                pd.Series(0.0, index=df.index))
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    w = np.array([weights[c] for c in cols], dtype=float)
    mask = ~np.isnan(vals)
    wsum = (mask * w[None, :]).sum(axis=1)
    num = np.nansum(np.where(mask, vals, 0.0) * w[None, :], axis=1)
    score = np.where(wsum > 0, num / np.where(wsum == 0, 1, wsum), np.nan)
    coverage = wsum / w.sum()
    return (pd.Series(score, index=df.index),
            pd.Series(coverage, index=df.index))


def score_universe(raw_blocks: Dict[str, pd.DataFrame], cfg: dict,
                   sectors: pd.Series | None = None) -> pd.DataFrame:
    """
    raw_blocks: {"technical": df, "momentum": df, ...} of raw sub-factor values.
    Returns a frame with one row per symbol: block scores, composite, rank,
    plus every diagnostic column the factor modules exported.
    """
    weight_key = {
        "technical": "technical_weights",
        "momentum": "momentum_weights",
        "fundamental": "fundamental_weights",
        "sentiment": "sentiment_weights",
        "trend": "trend_weights",
        "flows": "flows_weights",
    }

    index = None
    for df in raw_blocks.values():
        index = df.index if index is None else index.union(df.index)

    result = pd.DataFrame(index=index)
    coverages = {}
    diagnostics = pd.DataFrame(index=index)

    for block in BLOCKS:
        raw = raw_blocks.get(block)
        if raw is None or raw.empty:
            result[block] = np.nan
            coverages[block] = pd.Series(0.0, index=index)
            continue
        std = standardise(raw, cfg, sectors)
        diagnostics = diagnostics.join(
            std[[c for c in std.columns if c.startswith("_")]], how="left")
        sub_w = cfg.get(weight_key[block], {})
        if not sub_w:  # equal weights if unspecified
            sub_w = {c: 1.0 for c in std.columns if not c.startswith("_")}
        s, cov = blend(std, sub_w)
        result[block] = np.clip(s.reindex(index), -3, 3)
        coverages[block] = cov.reindex(index).fillna(0.0)
        # keep the standardised sub-factors for the "why" panel on the site
        for c in std.columns:
            if not c.startswith("_"):
                result[f"{block}__{c}"] = std[c].reindex(index)

    cov_df = pd.DataFrame(coverages)
    block_w = cfg["weights"]
    total_w = sum(block_w[b] for b in BLOCKS)

    # overall coverage weights each block's availability by that block's weight
    result["coverage"] = sum(cov_df[b] * block_w[b] for b in BLOCKS) / total_w

    comp, _ = blend(result[BLOCKS], {b: block_w[b] for b in BLOCKS})
    result["composite"] = comp

    min_cov = cfg["scoring"].get("min_factor_coverage", 0.6)
    result["eligible"] = result["coverage"] >= min_cov
    dropped = int((~result["eligible"]).sum())
    if dropped:
        log.info("%d symbols dropped for insufficient factor coverage (<%.0f%%)",
                 dropped, min_cov * 100)

    result = result.join(diagnostics, how="left")
    ranked = result[result["eligible"] & result["composite"].notna()].copy()
    ranked = ranked.sort_values("composite", ascending=False)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    # percentile within today's universe — easier to read than a raw z-score
    ranked["percentile"] = 100 * (1 - (ranked["rank"] - 1) / max(len(ranked), 1))
    return ranked


def explain(row: pd.Series, cfg: dict, top_k: int = 4) -> List[tuple[str, float]]:
    """
    Which sub-factors actually pushed this stock up the list?
    Contribution = block weight x sub-factor weight x standardised value.
    """
    weight_key = {
        "technical": "technical_weights", "momentum": "momentum_weights",
        "fundamental": "fundamental_weights", "sentiment": "sentiment_weights",
        "trend": "trend_weights",
        "flows": "flows_weights",
    }
    contribs = []
    total_bw = sum(cfg["weights"][b] for b in BLOCKS)
    for b in BLOCKS:
        bw = cfg["weights"][b] / total_bw
        sw = cfg.get(weight_key[b], {})
        swt = sum(sw.values()) or 1.0
        for name, w in sw.items():
            col = f"{b}__{name}"
            if col in row.index and pd.notna(row[col]):
                contribs.append((f"{b}.{name}", float(bw * (w / swt) * row[col])))
    contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return contribs[:top_k]


PRETTY = {
    "technical.trend_structure": "Price above all key moving averages (stage-2 uptrend)",
    "technical.adx_strength": "ADX confirms a genuine, directional trend",
    "technical.rsi_zone": "RSI in the strong-but-not-overbought zone",
    "technical.macd": "MACD histogram positive and expanding",
    "technical.bollinger": "Breaking out of a volatility squeeze",
    "technical.supertrend": "Supertrend in buy mode with room above the stop",
    "technical.volume_confirmation": "Volume and OBV confirm the price move",
    "technical.breakout_proximity": "Trading near its 52-week high / 20-day breakout",
    "technical.delivery_quality": "Rising delivery % — real buying, not intraday churn",
    "momentum.ret_1m": "Strong 1-month return",
    "momentum.ret_3m": "Strong 3-month return",
    "momentum.ret_6m": "Strong 6-month return",
    "momentum.ret_12m_1m": "Strong 12-month momentum (ex last month)",
    "momentum.rs_vs_market": "Outperforming the index and still widening the gap",
    "momentum.consistency": "Gains are steady rather than one lucky gap",
    "fundamental.valuation": "Cheap versus its own sector's multiples",
    "fundamental.quality": "Strong returns on capital, comfortable interest cover",
    "fundamental.growth": "Revenue and profit growing year on year",
    "fundamental.cash_quality": "Profit converts to real cash — positive free cash flow",
    "fundamental.ownership": "Healthy promoter holding, little or no pledging",
    "sentiment.announcements": "Positive corporate filings in recent weeks",
    "sentiment.deals": "Net institutional buying in bulk/block deals",
    "sentiment.results_reaction": "Market reacted well to the last results",
    "trend.historical_edge": "This exact setup has historically preceded gains in this stock",
    "trend.regime_fit": "Stock's behaviour is trend-persistent, suiting this signal",
    "trend.seasonality": "Seasonally favourable month for this stock",
    "trend.drawdown_health": "Close to its highs with a good recovery record",
    "trend.volatility_regime": "Volatility is calm rather than expanding",
    "flows.oi_buildup": "Futures open interest rising with price — a long build-up",
    "flows.pcr_position": "Options positioning is constructive (put writers comfortable)",
    "flows.basis": "Futures trading at a premium — leveraged money is paying to be long",
    "flows.institutional": "FIIs/DIIs increased their holding last quarter",
}


# The mirror image, for factors that are dragging the score down. Written out
# rather than derived, because "not X" reads badly and often means something
# different from the negation of X.
PRETTY_NEGATIVE = {
    "technical.trend_structure": "Below key moving averages — the trend structure is broken",
    "technical.adx_strength": "No real trend: ADX is flat or the sellers have the edge",
    "technical.rsi_zone": "RSI is either overbought or in washout territory",
    "technical.macd": "MACD momentum is negative or fading",
    "technical.bollinger": "Sitting in the lower half of its volatility band",
    "technical.supertrend": "Supertrend is in sell mode or price is close to flipping it",
    "technical.volume_confirmation": "Volume is not confirming the price move",
    "technical.breakout_proximity": "Well below its 52-week high — no breakout here",
    "technical.delivery_quality": "Delivery percentage is low or falling — weak hands",
    "momentum.ret_1m": "Weak 1-month return",
    "momentum.ret_3m": "Weak 3-month return",
    "momentum.ret_6m": "Weak 6-month return",
    "momentum.ret_12m_1m": "Weak 12-month momentum",
    "momentum.rs_vs_market": "Lagging the index rather than leading it",
    "momentum.consistency": "Gains are lumpy — driven by a few large moves",
    "fundamental.valuation": "Expensive relative to its own sector",
    "fundamental.quality": "Weak returns on capital, thin interest cover or heavy debt",
    "fundamental.growth": "Revenue and profit growth are soft",
    "fundamental.cash_quality": "Reported profit is not converting into cash",
    "fundamental.ownership": "Low promoter holding or pledged shares",
    "sentiment.announcements": "Recent corporate filings skew negative",
    "sentiment.deals": "Net institutional selling in bulk/block deals",
    "sentiment.results_reaction": "The market sold the last set of results",
    "trend.historical_edge": "Historically this setup has NOT worked in this stock",
    "trend.regime_fit": "This stock mean-reverts — trend signals fit it poorly",
    "trend.seasonality": "Seasonally a weak month for this stock",
    "trend.drawdown_health": "Deep in a drawdown with a poor recovery record",
    "trend.volatility_regime": "Volatility is expanding — often distribution, not accumulation",
    "flows.oi_buildup": "The move is short covering or a short build-up, not fresh buying",
    "flows.pcr_position": "Heavy call writing overhead — options sellers are capping it",
    "flows.basis": "Futures at a discount to spot — derivatives aren't backing this move",
    "flows.institutional": "Institutions trimmed their stake, or pledging went up",
}


def humanise(contribs: List[tuple[str, float]]) -> List[str]:
    out = []
    for name, val in contribs:
        if val < 0:
            label = PRETTY_NEGATIVE.get(name)
            if label is None:
                label = "Drag from " + PRETTY.get(name, name.replace("_", " ")).lower()
            out.append("Against it: " + label[0].lower() + label[1:])
        else:
            out.append(PRETTY.get(name, name.replace("_", " ")))
    return out
