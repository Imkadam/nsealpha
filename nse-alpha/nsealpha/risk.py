"""
risk.py — the part that tells you when NOT to buy, and how much.

Two jobs:

1. MARKET REGIME. A stock-ranking model always produces a top 10, including on
   the morning of a crash. Ranking is relative; survival is absolute. So before
   any pick is shown we assess the market itself:
     - Is the benchmark above its 200-DMA? (the single most useful risk switch
       there is: most large drawdowns happen below it)
     - Is it above its 50-DMA?
     - Market breadth: what share of the universe is above its own 50-DMA?
     - India VIX level.
   These produce a RISK-ON / CAUTION / RISK-OFF verdict and a suggested exposure
   multiplier. The picks are still shown in risk-off — but with a clear warning
   and a smaller suggested position size.

2. POSITION SIZING & LEVELS. For each pick:
     stop      entry - (atr_stop_multiple x ATR14), i.e. sized to the stock's own
               normal daily travel rather than an arbitrary 5%
     targets   R-multiples of the risk taken (default 2R and 3.5R)
     size      shares such that (entry - stop) x shares = account_risk_pct of
               capital. This is the "risk 1% per trade" rule, which is what keeps
               a losing streak survivable.
   Plus sector caps so the top 10 is not seven cement companies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Regime:
    verdict: str            # RISK-ON | CAUTION | RISK-OFF
    exposure: float         # 0.0 - 1.0 suggested exposure multiplier
    notes: List[str]
    benchmark_above_200: Optional[bool] = None
    benchmark_above_50: Optional[bool] = None
    breadth_pct: Optional[float] = None
    vix: Optional[float] = None
    bench_ret_1m: Optional[float] = None
    fii_20d_cr: Optional[float] = None
    dii_20d_cr: Optional[float] = None
    fii_long_pct: Optional[float] = None


def assess_regime(benchmark: pd.DataFrame, snapshot: pd.DataFrame,
                  cfg: dict, vix: Optional[float] = None,
                  flows: Optional[Dict[str, float]] = None) -> Regime:
    notes: List[str] = []
    score = 0
    above200 = above50 = None
    ret1m = None

    if benchmark is not None and len(benchmark) > 210:
        b = benchmark.sort_values("date")["close"].reset_index(drop=True)
        sma200 = b.rolling(200).mean().iloc[-1]
        sma50 = b.rolling(50).mean().iloc[-1]
        last = b.iloc[-1]
        above200 = bool(last > sma200)
        above50 = bool(last > sma50)
        ret1m = float(last / b.iloc[-22] - 1) * 100 if len(b) > 22 else None

        if above200:
            score += 2
            notes.append(f"Benchmark is {100*(last/sma200-1):.1f}% above its 200-DMA — "
                         "the primary trend is intact.")
        else:
            score -= 3
            notes.append(f"Benchmark is {100*(last/sma200-1):.1f}% BELOW its 200-DMA — "
                         "most large drawdowns happen here. Treat every long with suspicion.")
        if above50:
            score += 1
            notes.append("Benchmark is above its 50-DMA (intermediate trend up).")
        else:
            score -= 1
            notes.append("Benchmark is below its 50-DMA (intermediate trend down).")

    breadth = None
    if snapshot is not None and not snapshot.empty and "sma50" in snapshot.columns:
        ok = (snapshot["close"] > snapshot["sma50"])
        breadth = float(ok.mean() * 100)
        if breadth >= 60:
            score += 2
            notes.append(f"Breadth is healthy — {breadth:.0f}% of the universe is above its 50-DMA.")
        elif breadth >= 45:
            notes.append(f"Breadth is mixed — {breadth:.0f}% of stocks above their 50-DMA.")
        else:
            score -= 2
            notes.append(f"Breadth is weak — only {breadth:.0f}% of stocks are above their "
                         "50-DMA. Rallies without breadth tend to fail.")

    if vix is not None and not np.isnan(vix):
        if vix >= cfg["risk"]["vix_risk_off"]:
            score -= 2
            notes.append(f"India VIX at {vix:.1f} — elevated fear, wider stops needed.")
        elif vix >= cfg["risk"]["vix_caution"]:
            score -= 1
            notes.append(f"India VIX at {vix:.1f} — above the calm-market range.")
        else:
            score += 1
            notes.append(f"India VIX at {vix:.1f} — volatility is contained.")

    # --- institutional flow ------------------------------------------------
    # FIIs move the Indian index; DIIs, fed by domestic SIP money, increasingly
    # absorb what FIIs sell. The divergence between the two says more than either
    # number alone: heavy FII selling that DIIs fully absorb is a very different
    # market from heavy FII selling into no bid.
    fii20 = dii20 = fii_long = None
    if flows:
        fii20 = flows.get("fii_net_20d_cr")
        dii20 = flows.get("dii_net_20d_cr")
        fii_long = flows.get("fii_index_fut_long_pct")

        if fii20 is not None:
            if fii20 > 15000:
                score += 2
                notes.append(f"FIIs bought Rs {fii20:,.0f} cr net over 20 sessions — "
                             "foreign money is coming in.")
            elif fii20 < -15000:
                score -= 2
                absorbed = dii20 is not None and dii20 > abs(fii20) * 0.8
                notes.append(
                    f"FIIs sold Rs {abs(fii20):,.0f} cr net over 20 sessions"
                    + (f", but DIIs absorbed Rs {dii20:,.0f} cr of it — domestic "
                       "flows are holding the market up." if absorbed
                       else " with limited domestic absorption."))
                if absorbed:
                    score += 1
            else:
                notes.append(f"FII net flow is roughly balanced "
                             f"(Rs {fii20:,.0f} cr over 20 sessions).")

        if fii_long is not None:
            if fii_long < 30:
                score -= 1
                notes.append(f"FIIs are only {fii_long:.0f}% long in index futures — "
                             "defensively positioned.")
            elif fii_long > 65:
                score += 1
                notes.append(f"FIIs are {fii_long:.0f}% long in index futures — "
                             "positioned for upside.")

    if score >= 3:
        verdict, exposure = "RISK-ON", 1.0
    elif score >= 0:
        verdict, exposure = "CAUTION", 0.6
    else:
        verdict, exposure = "RISK-OFF", 0.25
        notes.append("Suggested action: hold cash, cut position sizes, or wait for the "
                     "benchmark to reclaim its 200-DMA. The ranking below is still "
                     "relative — the best stock in a falling market can still fall.")

    return Regime(verdict=verdict, exposure=exposure, notes=notes,
                  benchmark_above_200=above200, benchmark_above_50=above50,
                  breadth_pct=breadth, vix=vix, bench_ret_1m=ret1m,
                  fii_20d_cr=fii20, dii_20d_cr=dii20, fii_long_pct=fii_long)


def trade_plan(row: pd.Series, cfg: dict, capital: float = 100000.0,
               exposure: float = 1.0) -> Dict[str, float]:
    """
    ATR-based levels and a risk-parity position size for one pick.
    `row` needs: close, atr14, sma20, sma50, hi_52w.
    """
    entry = float(row.get("close", np.nan))
    atr = float(row.get("atr14", np.nan))
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return {}

    k = cfg["risk"]["atr_stop_multiple"]
    stop = entry - k * atr

    # If a moving average the whole market watches sits just *below* the ATR stop,
    # slide the stop under it — price routinely wicks through a level and recovers,
    # and being stopped out one rupee above support is the most annoying way to lose.
    # Only do this when the level is within one ATR, so the stop stays tight.
    for level in ("sma20", "sma50"):
        lv = row.get(level, np.nan)
        if np.isfinite(lv) and stop < lv < entry and (stop - lv) > -atr:
            stop = float(lv) * 0.995
            break

    # A stop further than max_stop_pct away is not risk management, it is hope.
    max_stop_pct = cfg["risk"].get("max_stop_pct", 12.0)
    stop = max(stop, entry * (1 - max_stop_pct / 100.0))

    risk_per_share = max(entry - stop, atr * 0.5)
    t1, t2 = cfg["risk"]["target_multiples"]
    risk_amt = capital * (cfg["risk"]["account_risk_pct"] / 100.0) * exposure
    shares = int(max(risk_amt / risk_per_share, 0))

    return {
        "entry": round(entry, 2),
        "entry_zone_low": round(entry - 0.35 * atr, 2),
        "entry_zone_high": round(entry + 0.35 * atr, 2),
        "stop": round(stop, 2),
        "stop_pct": round(100 * (stop / entry - 1), 2),
        "target1": round(entry + t1 * risk_per_share, 2),
        "target2": round(entry + t2 * risk_per_share, 2),
        "target1_pct": round(100 * (t1 * risk_per_share) / entry, 1),
        "target2_pct": round(100 * (t2 * risk_per_share) / entry, 1),
        "risk_per_share": round(risk_per_share, 2),
        "shares": shares,
        "capital_at_risk": round(shares * risk_per_share, 0),
        "position_value": round(shares * entry, 0),
        "atr_pct": round(100 * atr / entry, 2),
    }


def apply_portfolio_caps(ranked: pd.DataFrame, sectors: Dict[str, str],
                         cfg: dict) -> pd.DataFrame:
    """
    Walk the ranking top-down, skipping names that would breach the per-sector cap.
    Ten stocks from one sector is one position with extra brokerage.
    """
    max_per_sector = cfg["risk"].get("max_per_sector", 3)
    top_n = cfg["scoring"].get("top_n", 10)

    # If we barely know anyone's sector (fundamentals not fetched yet), applying
    # the cap would silently truncate the list to `max_per_sector` names, all
    # bucketed as "Unclassified". Better to return a full list and say so.
    known = sum(1 for s in ranked.index if sectors.get(s))
    coverage = known / max(len(ranked), 1)
    if coverage < 0.5:
        import logging
        logging.getLogger("nsealpha.risk").warning(
            "sector known for only %.0f%% of ranked stocks — skipping the "
            "per-sector cap this run (fetch fundamentals to enable it)",
            coverage * 100)
        max_per_sector = top_n

    counts: Dict[str, int] = {}
    picked, skipped = [], []
    for sym, row in ranked.iterrows():
        sec = sectors.get(sym) or "Unclassified"
        if counts.get(sec, 0) >= max_per_sector:
            skipped.append((sym, sec))
            continue
        counts[sec] = counts.get(sec, 0) + 1
        picked.append(sym)
        if len(picked) >= top_n:
            break
    out = ranked.loc[picked].copy()
    out["sector"] = [sectors.get(s) or "Unclassified" for s in out.index]
    out["final_rank"] = np.arange(1, len(out) + 1)
    return out


def red_flags(row: pd.Series) -> List[str]:
    """Things a human should look at before clicking buy, even on a top pick."""
    flags = []
    if row.get("rsi14", 50) > 78:
        flags.append("RSI above 78 — overbought, a pullback is likely before continuation")
    if row.get("atr_pct", 0) > 6:
        flags.append(f"Very volatile ({row.get('atr_pct'):.1f}% daily ATR) — size down")
    if row.get("pct_from_52w_high", -50) < -30:
        flags.append("More than 30% below its 52-week high — this is a bounce, not a breakout")
    if row.get("turnover_20d_cr", 99) < 15:
        flags.append(f"Thin liquidity (Rs {row.get('turnover_20d_cr', 0):.0f} cr/day) — "
                     "use limit orders, expect slippage")
    if row.get("_pledge", 0) and row.get("_pledge", 0) > 10:
        flags.append(f"Promoter pledge at {row.get('_pledge'):.0f}%")
    if row.get("_de", 0) and row.get("_de", 0) > 1.5:
        flags.append(f"High leverage (D/E {row.get('_de'):.1f}x)")
    # The guide's checklist thresholds, surfaced as flags even when the stock
    # still ranks well on everything else.
    icr = row.get("_icr")
    if icr is not None and np.isfinite(icr) and icr < 4.0:
        flags.append(f"Interest coverage only {icr:.1f}x — the guide's floor is 4.0x")
    conv = row.get("_cash_conversion")
    if conv is not None and np.isfinite(conv) and conv < 0.8:
        flags.append(f"Cash conversion {conv:.2f}x — reported profit is not turning "
                     "into operating cash")
    fcf_cum = row.get("_fcf_cumulative")
    if fcf_cum is not None and np.isfinite(fcf_cum) and fcf_cum < 0:
        yrs = row.get("_fcf_years")
        span = f" over {int(yrs)} years" if yrs and np.isfinite(yrs) else ""
        flags.append(f"Cumulative free cash flow is negative{span}")
    # Only flag a filing that is both SEVERE and RECENT, and name it. A flag that
    # fires on nine picks out of ten trains you to ignore all of them.
    worst = row.get("ann_worst")
    age = row.get("_ann_worst_age")
    if (worst is not None and np.isfinite(worst) and worst <= -3.0
            and (age is None or not np.isfinite(age) or age <= 21)):
        subj = row.get("_ann_worst_subject")
        when = f" ({int(age)}d ago)" if age is not None and np.isfinite(age) else ""
        if isinstance(subj, str) and subj.strip():
            flags.append(f'Negative filing{when}: "{subj[:90]}"')
        else:
            flags.append(f"A serious negative filing was made{when} — read it")
    if row.get("_edge_n", 0) and row.get("_edge_n", 0) < 25:
        flags.append("Thin historical sample for this setup — lower confidence")
    if row.get("vol_surge", 1) > 3.5:
        flags.append("Volume 3.5x its average — check for news or a block deal before entering")
    basis = row.get("_basis_pct")
    if basis is not None and np.isfinite(basis) and basis < -0.6:
        flags.append(f"Futures at a {abs(basis):.1f}% discount to spot — derivatives "
                     "are not backing this move")
    pledge_chg = row.get("_pledge_chg")
    if pledge_chg is not None and np.isfinite(pledge_chg) and pledge_chg > 2:
        flags.append(f"Promoter pledging rose {pledge_chg:.1f} points last quarter")
    return flags
