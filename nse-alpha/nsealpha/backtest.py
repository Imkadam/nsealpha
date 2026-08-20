"""
backtest.py — walk-forward validation of the whole ranking, not just one factor.

The rule this file exists to enforce: **the backtest calls the same `pipeline.run`
the live screen calls.** A backtest written against a reimplementation of the
model is a story about a model that doesn't exist.

Mechanics:
  - Step through history every `rebalance_days` trading days.
  - At each step, re-run the full pipeline with `as_of` set to that date, so every
    indicator, factor and filter only ever sees data available then.
  - Buy the top N equally weighted, hold until the next step, charge `cost_bps`
    round-trip.
  - Compare against the benchmark over the same window.

Honest caveats, stated because they matter more than the headline number:
  * Fundamentals are stored as a current snapshot, not as point-in-time history.
    NSE does not publish a clean historical ratio archive, so the fundamental
    block in a backtest carries look-ahead bias. Set `weights.fundamental: 0` in
    the config when you want a clean read, then compare.
  * Announcements only go back as far as you have ingested them.
  * Survivorship: the bhavcopy archive contains delisted names on the days they
    traded, so this is better than most retail backtests, but a stock that
    delisted before your first bhavcopy download is simply absent.
  * Costs of 35bps round trip assume liquid names and limit orders. Widen it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .pipeline import run as run_pipeline
from .store import Store

log = logging.getLogger("nsealpha.backtest")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    periods: pd.DataFrame
    stats: Dict[str, float]
    summary: str


def _forward_return(prices_wide: pd.DataFrame, sym: str,
                    d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    if sym not in prices_wide.columns:
        return np.nan
    s = prices_wide[sym]
    try:
        p0 = s.asof(d0)
        p1 = s.asof(d1)
    except Exception:
        return np.nan
    if not np.isfinite(p0) or not np.isfinite(p1) or p0 <= 0:
        return np.nan
    return float(p1 / p0 - 1)


def run_backtest(cfg: dict, store: Store, start: Optional[str] = None,
                 max_periods: int = 40, verbose: bool = True,
                 progress=None) -> BacktestResult:
    """
    `progress` is an optional callable (done, total, label) -> None. A walk-forward
    backtest re-runs the whole model per period and takes minutes; a UI that shows
    nothing for four minutes is indistinguishable from one that has hung.
    """
    bcfg = cfg["backtest"]
    start = start or bcfg["start"]
    step = int(bcfg["rebalance_days"])
    top_n = int(bcfg["top_n"])
    cost = bcfg["cost_bps"] / 10000.0

    prices = store.load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    all_dates = np.array(sorted(prices["date"].unique()))
    start_ts = pd.Timestamp(start)
    tradable = all_dates[all_dates >= start_ts]
    if len(tradable) < step * 3:
        raise RuntimeError("Not enough history after the backtest start date.")

    # Rebalance dates, capped so a backtest doesn't take all afternoon. When the
    # cap bites we keep the most recent CONTIGUOUS periods rather than sampling
    # across the range — compounding disjoint windows and calling the product a
    # "total return" would be nonsense.
    idxs = list(range(0, len(tradable) - step, step))
    if len(idxs) > max_periods:
        idxs = idxs[-max_periods:]
        log.info("capped to the most recent %d periods (from %s)",
                 max_periods, pd.Timestamp(tradable[idxs[0]]).date())

    wide = prices.pivot_table(index="date", columns="symbol", values="close").sort_index()

    bench = store.load_index(cfg["risk"]["benchmark"])
    if bench.empty:
        bench_series = wide.pct_change().mean(axis=1).add(1).cumprod()
    else:
        bench_series = bench.set_index("date")["close"].sort_index()

    trades, periods = [], []
    for n, i in enumerate(idxs):
        d0 = pd.Timestamp(tradable[i])
        d1 = pd.Timestamp(tradable[min(i + step, len(tradable) - 1)])
        if verbose:
            print(f"  period {n+1}/{len(idxs)}  {d0.date()} → {d1.date()}   ",
                  end="\r", flush=True)
        if progress:
            progress(n, len(idxs), f"{d0.date()} → {d1.date()}")
        try:
            res = run_pipeline(cfg, store, as_of=d0, verbose=False)
        except Exception as e:
            log.debug("period %s skipped: %s", d0.date(), e)
            continue
        picks = res.picks.head(top_n)
        if picks.empty:
            continue

        rets = []
        for sym in picks.index:
            r = _forward_return(wide, sym, d0, d1)
            if np.isfinite(r):
                rets.append(r)
                trades.append({"date": d0, "exit": d1, "symbol": sym,
                               "rank": int(picks.loc[sym, "final_rank"]),
                               "composite": float(picks.loc[sym, "composite"]),
                               "ret": r})
        if not rets:
            continue

        port = float(np.mean(rets)) - cost
        try:
            b0, b1 = bench_series.asof(d0), bench_series.asof(d1)
            bret = float(b1 / b0 - 1) if np.isfinite(b0) and np.isfinite(b1) else np.nan
        except Exception:
            bret = np.nan
        periods.append({"date": d0, "exit": d1, "n": len(rets),
                        "portfolio": port, "benchmark": bret,
                        "excess": port - bret if np.isfinite(bret) else np.nan,
                        "win_rate": float(np.mean([r > 0 for r in rets])),
                        "best": max(rets), "worst": min(rets)})
    if verbose:
        print(" " * 60, end="\r")
    if progress:
        progress(len(idxs), len(idxs), "done")

    tdf = pd.DataFrame(trades)
    pdf = pd.DataFrame(periods)
    if pdf.empty:
        raise RuntimeError("Backtest produced no periods — check history and start date.")

    eq = (1 + pdf["portfolio"]).cumprod()
    beq = (1 + pdf["benchmark"].fillna(0)).cumprod()
    per_year = 252 / step
    n_years = len(pdf) / per_year

    cagr = float(eq.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)
    bcagr = float(beq.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)
    vol = float(pdf["portfolio"].std() * np.sqrt(per_year))
    sharpe = float((pdf["portfolio"].mean() * per_year) / vol) if vol > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    hit = float((pdf["portfolio"] > 0).mean())
    beat = float((pdf["excess"] > 0).mean())

    # Annualising a handful of five-day periods produces numbers like "180%
    # annualised, Sharpe 5.7" from what is really three coin flips. Rather than
    # print those and hope the reader discounts them, mark the run as unreliable
    # and let the caller suppress them.
    MIN_RELIABLE_PERIODS = 12
    reliable = len(pdf) >= MIN_RELIABLE_PERIODS

    stats = {
        "periods": int(len(pdf)),
        "reliable": bool(reliable),
        "min_reliable_periods": MIN_RELIABLE_PERIODS,
        "hold_days": step,
        "total_return_pct": float(eq.iloc[-1] - 1) * 100,
        "benchmark_return_pct": float(beq.iloc[-1] - 1) * 100,
        "cagr_pct": cagr * 100,
        "benchmark_cagr_pct": bcagr * 100,
        "volatility_pct": vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": dd * 100,
        "period_win_rate_pct": hit * 100,
        "beat_benchmark_pct": beat * 100,
        "avg_stock_win_rate_pct": float(pdf["win_rate"].mean()) * 100,
        "trades": int(len(tdf)),
    }

    if reliable:
        summary = (
            f"{stats['periods']} rebalances at a {step}-day hold. "
            f"Strategy {stats['total_return_pct']:.1f}% total "
            f"({stats['cagr_pct']:.1f}% annualised) vs benchmark "
            f"{stats['benchmark_return_pct']:.1f}% "
            f"({stats['benchmark_cagr_pct']:.1f}% annualised). "
            f"Sharpe {stats['sharpe']:.2f}, max drawdown "
            f"{stats['max_drawdown_pct']:.1f}%, "
            f"{stats['period_win_rate_pct']:.0f}% of periods positive, "
            f"beat the benchmark in {stats['beat_benchmark_pct']:.0f}% of periods. "
            f"Fundamentals are a current snapshot, so treat the fundamental "
            f"block's contribution here as optimistic."
        )
    else:
        summary = (
            f"Only {stats['periods']} rebalances at a {step}-day hold — too few to "
            f"annualise or compute a meaningful Sharpe, so those are withheld. "
            f"Over the window tested the strategy returned "
            f"{stats['total_return_pct']:.1f}% against the benchmark's "
            f"{stats['benchmark_return_pct']:.1f}%, and beat it in "
            f"{stats['beat_benchmark_pct']:.0f}% of periods. Re-run with at least "
            f"{MIN_RELIABLE_PERIODS} periods before drawing any conclusion."
        )
    return BacktestResult(trades=tdf, periods=pdf, stats=stats, summary=summary)
