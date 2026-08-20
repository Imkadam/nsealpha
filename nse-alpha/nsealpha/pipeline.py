"""
pipeline.py — the orchestrator. Everything else is a component; this is the wiring.

    load config -> refresh data -> filter universe -> compute indicators
    -> run the five factor blocks -> standardise & blend -> apply risk overlay
    -> build trade plans -> persist -> render the website

`run()` is what run_daily.py calls. It is also what backtest.py calls with an
`as_of` date, which is how we make sure the backtest and the live run share one
code path — a backtest that uses different code from production is a fairy tale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from . import adjust, financials, flows, panel, risk, scoring
from .factors import FactorContext
from .factors import flows as f_flows
from .factors import fundamental as f_fund
from .factors import momentum as f_mom
from .factors import sentiment as f_sent
from .factors import technical as f_tech
from .factors import trend as f_trend
from .store import Store
from .universe import build_universe, rejection_summary

log = logging.getLogger("nsealpha.pipeline")


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass
class RunResult:
    as_of: pd.Timestamp
    picks: pd.DataFrame          # top N with trade plans
    ranked: pd.DataFrame         # full ranking
    regime: risk.Regime
    universe_size: int
    rejected: pd.DataFrame
    sectors: Dict[str, str]
    snapshot: pd.DataFrame
    config: dict
    ca_events: pd.DataFrame = None      # corporate actions applied to history
    market_flows: Dict[str, float] = None


# ---------------------------------------------------------------- data refresh

def refresh_data(cfg: dict, store: Store, full_backfill: bool = False,
                 skip_fundamentals: bool = False, verbose: bool = True) -> None:
    """Pull anything we're missing from NSE. Safe to call every day."""
    from .nseclient import NSEClient
    from . import ingest

    d = cfg["data"]
    client = NSEClient(cache_dir=d["cache_dir"],
                       requests_per_second=d["requests_per_second"],
                       timeout=d["request_timeout"],
                       max_retries=d["max_retries"],
                       server_mode=d.get("server_mode", False))

    def prog(i, n, cur):
        if verbose:
            print(f"  [{i}/{n}] {cur}", end="\r", flush=True)

    days = d["history_days"] if full_backfill else 25
    log.info("fetching bhavcopies (%d days back)...", days)
    ingest.backfill_prices(client, store, days_back=days, progress=prog)
    log.info("fetching index history...")
    ingest.backfill_indices(client, store, days_back=days)

    prices = store.load_prices()
    if not prices.empty:
        ingest.build_security_master(client, store, prices)

    log.info("fetching corporate announcements...")
    try:
        ingest.ingest_announcements(client, store, days_back=60)
    except Exception as e:
        log.warning("announcements unavailable: %s", e)
    try:
        ingest.ingest_deals(client, store, days_back=45)
    except Exception as e:
        log.warning("bulk/block deals unavailable: %s", e)

    # Corporate actions: needed to adjust history for splits and bonuses. Always
    # fetch these, even on a light refresh — a missed split corrupts the series.
    log.info("fetching corporate actions (splits, bonuses, rights)...")
    try:
        n = ingest.ingest_corporate_actions(
            client, store, days_back=d["history_days"] if full_backfill else 90)
        log.info("corporate actions on file: %d rows", n)
    except Exception as e:
        log.warning("corporate actions unavailable: %s — history will NOT be "
                    "split-adjusted from this source this run", e)

    # ---- flow data ---------------------------------------------------------
    if d.get("fetch_flows", True):
        log.info("fetching F&O bhavcopies (open interest, PCR, basis)...")
        try:
            flows.backfill_fno(client, store,
                               days_back=min(days, 400), progress=prog)
        except Exception as e:
            log.warning("F&O data unavailable: %s", e)
        log.info("fetching FII/DII activity and participant-wise OI...")
        try:
            flows.ingest_fii_dii(client, store)
        except Exception as e:
            log.warning("FII/DII activity unavailable: %s", e)
        try:
            flows.ingest_participant_oi(client, store,
                                        days_back=200 if full_backfill else 20)
        except Exception as e:
            log.warning("participant OI unavailable: %s", e)

    if not skip_fundamentals and not prices.empty:
        # only for names that survive the filters — no point fetching ratios for
        # 1200 illiquid shells
        u = build_universe(prices, cfg, securities=store.load_securities())
        log.info("refreshing fundamentals for %d symbols (this is the slow part)...",
                 len(u.symbols))
        ingest.refresh_fundamentals(client, store, u.symbols,
                                    max_age_days=7, progress=prog)
        # Shareholding is quarterly data — monthly refresh is plenty.
        stale = store.get_meta("shareholding_refreshed", "1970-01-01")
        if (date.today() - datetime.strptime(stale, "%Y-%m-%d").date()).days >= 30:
            log.info("refreshing quarterly shareholding patterns...")
            try:
                flows.ingest_shareholding(client, store, u.symbols, progress=prog)
                store.set_meta("shareholding_refreshed", date.today().isoformat())
            except Exception as e:
                log.warning("shareholding patterns unavailable: %s", e)

        # Annual statements change at most four times a year — monthly is ample,
        # and this is the slowest single pass in the whole refresh.
        log.info("refreshing annual financial statements "
                 "(interest coverage, free cash flow, cash conversion)...")
        try:
            financials.refresh_statements(store, u.symbols, max_age_days=30,
                                          progress=prog)
        except Exception as e:
            log.warning("financial statements unavailable: %s — interest coverage, "
                        "FCF and cash conversion will score as missing", e)
    client.close()


# --------------------------------------------------------------------- the run
#
# The run is split in two on purpose:
#
#   prepare()             everything expensive and weight-independent — loading
#                         prices, adjusting for corporate actions, building
#                         indicators for ~2000 symbols, and computing the raw
#                         sub-factor values. Takes a minute or two.
#   score_from_prepared() everything that depends on the weights — winsorise,
#                         z-score, blend, rank, size positions. Takes ~50ms.
#
# That split is what lets the Streamlit app put the factor weights on sliders and
# re-rank the whole market instantly: prepare() is cached, and only the cheap half
# re-runs when you move a slider. run() below is just the two called back to back,
# so the CLI and the backtest are unaffected.

@dataclass
class PreparedRun:
    """Weight-independent state. Cache this; re-score from it as often as you like."""
    as_of: pd.Timestamp
    frames: Dict[str, pd.DataFrame]
    snapshot: pd.DataFrame
    raw_blocks: Dict[str, pd.DataFrame]
    benchmark: pd.DataFrame
    sectors: Dict[str, str]
    universe_symbols: List[str]
    rejected: pd.DataFrame
    ca_events: pd.DataFrame
    market_flows: Dict[str, float]
    fno: pd.DataFrame
    shareholding: pd.DataFrame
    fundamentals: pd.DataFrame
    financials: pd.DataFrame
    announcements: pd.DataFrame
    vix: Optional[float]
    prices: pd.DataFrame


def prepare(cfg: dict, store: Store, as_of: Optional[pd.Timestamp] = None,
            verbose: bool = True) -> PreparedRun:
    prices = store.load_prices()
    if prices.empty:
        raise RuntimeError("No price data. Run with --refresh first.")

    prices["date"] = pd.to_datetime(prices["date"])
    if as_of is not None:
        prices = prices[prices["date"] <= as_of]
    as_of = panel.latest_trading_date(prices)
    log.info("scoring as of %s", as_of.date())

    # ---- corporate actions -------------------------------------------------
    # Do this FIRST, before anything computes a return. An unadjusted split is a
    # -80% day as far as every downstream factor is concerned.
    ca_events = pd.DataFrame()
    if cfg["data"].get("adjust_corporate_actions", True):
        prices, events = adjust.adjusted_view(
            prices, store.load_corporate_actions(), enabled=True)
        ca_events = adjust.events_frame(events)
        if not ca_events.empty:
            recent = ca_events[ca_events["date"] >= as_of - pd.Timedelta(days=400)]
            if verbose and len(recent):
                print(f"  adjusted history for {len(recent)} corporate actions "
                      f"({recent['symbol'].nunique()} stocks)")
            bad = ca_events[ca_events["disagrees"]]
            if len(bad):
                log.warning("%d corporate actions where the two sources disagree — "
                            "see the site's data-quality panel", len(bad))

    securities = store.load_securities()

    index_members = None
    if cfg["universe"]["source"].upper() != "ALL":
        index_members = store.get_meta(f"members::{cfg['universe']['source']}")
        if not index_members:
            log.warning("no cached constituents for %s — using ALL",
                        cfg["universe"]["source"])

    u = build_universe(prices, cfg, index_members=index_members, securities=securities)
    if verbose:
        print(f"  universe: {len(u.symbols)} tradable names "
              f"({len(u.rejected)} filtered out)")
    if not u.symbols:
        raise RuntimeError("Every stock was filtered out — loosen filters in config.yaml")

    def prog(i, n, cur):
        if verbose:
            print(f"  indicators [{i}/{n}] {cur}      ", end="\r", flush=True)

    frames = panel.build_frames(prices, u.symbols, as_of=as_of, progress=prog)
    if verbose:
        print(" " * 60, end="\r")
    snapshot = panel.build_snapshot(frames)

    bench_name = cfg["risk"]["benchmark"]
    benchmark = store.load_index(bench_name)
    if benchmark.empty:
        # fall back to an equal-weight proxy built from the universe itself
        log.warning("no index data for %s — using an equal-weighted proxy", bench_name)
        benchmark = _proxy_benchmark(prices, u.symbols)
    benchmark = benchmark[pd.to_datetime(benchmark["date"]) <= as_of]

    fundamentals = store.load_fundamentals()
    sectors = panel.sector_map_from_fundamentals(fundamentals)
    sector_series = pd.Series(sectors).reindex(snapshot.index)

    # Statement-derived metrics: interest coverage, free cash flow, cash
    # conversion, gross margin, real ROCE. The sector map is passed in because
    # banks and NBFCs must be excluded from the interest and capex metrics.
    fin_metrics = financials.metrics_table(store, sector_map=sectors)
    if verbose and not fin_metrics.empty:
        covered = len(set(fin_metrics.index) & set(snapshot.index))
        print(f"  financial statements for {covered} of {len(snapshot)} stocks")

    since = (as_of - pd.Timedelta(days=75)).strftime("%Y-%m-%d")
    fno = flows.fno_features(store, as_of)
    holding = flows.shareholding_trend(store)
    mflows = flows.market_flow_summary(store, as_of)
    if verbose and not fno.empty:
        covered = len(set(fno.index) & set(snapshot.index))
        print(f"  derivatives data for {covered} of {len(snapshot)} stocks")

    ctx = FactorContext(
        frames=frames, snapshot=snapshot, benchmark=benchmark,
        sector_map=sectors, fundamentals=fundamentals,
        financials=fin_metrics,
        announcements=store.load_announcements(since),
        deals=store.load_deals(since),
        fno=fno, shareholding=holding, market_flows=mflows,
        config=cfg, as_of=as_of,
    )

    if verbose:
        print("  computing factor blocks...")
    raw_blocks = {
        "technical": f_tech.compute(ctx),
        "momentum": f_mom.compute(ctx),
        "fundamental": f_fund.compute(ctx),
        "sentiment": f_sent.compute(ctx),
        "trend": f_trend.compute(ctx),
        "flows": f_flows.compute(ctx),
    }

    # Normalise the numeric sub-factors to plain float64 once, here. Factor
    # modules can hand back object-dtype columns when a fundamentals field came
    # through as a string, and scoring would then silently coerce it on every
    # call. (Measured: no meaningful speed difference — this is for correctness
    # and predictable dtypes, not performance.)
    for block, df in raw_blocks.items():
        num = [c for c in df.columns if not c.startswith("_")]
        if num:
            raw_blocks[block] = df.astype({c: "float64" for c in num}, errors="ignore")

    return PreparedRun(
        as_of=as_of, frames=frames, snapshot=snapshot, raw_blocks=raw_blocks,
        benchmark=benchmark, sectors=sectors, universe_symbols=u.symbols,
        rejected=u.rejected, ca_events=ca_events, market_flows=mflows,
        fno=fno, shareholding=holding, fundamentals=fundamentals,
        financials=fin_metrics,
        announcements=ctx.announcements, vix=_latest_vix(store), prices=prices,
    )


def score_from_prepared(prep: PreparedRun, cfg: dict,
                        capital: float = 100000.0) -> RunResult:
    """The cheap half: apply weights, rank, size. Safe to call on every keystroke."""
    snapshot = prep.snapshot
    raw_blocks = prep.raw_blocks
    sector_series = pd.Series(prep.sectors).reindex(snapshot.index)

    ranked = scoring.score_universe(raw_blocks, cfg, sectors=sector_series)

    # attach the raw market data each pick needs for its trade plan and flags
    carry = [c for c in ["close", "atr14", "atr_pct", "rsi14", "adx", "sma20", "sma50",
                         "sma200", "hi_52w", "pct_from_52w_high", "turnover_20d_cr",
                         "vol_surge", "deliv_ma20", "deliv_pct", "ret_1m", "ret_3m",
                         "ret_6m", "ret_12m_1m", "supertrend_dir", "bb_pct_b",
                         "macd_hist", "vol20", "volume"] if c in snapshot.columns]
    ranked = ranked.join(snapshot[carry], how="left")
    if "announcements" in raw_blocks["sentiment"].columns:
        for c in ("ann_count", "ann_best", "ann_worst",
                  "_ann_worst_subject", "_ann_worst_age"):
            if c in raw_blocks["sentiment"].columns:
                ranked[c] = raw_blocks["sentiment"][c].reindex(ranked.index)
    # diagnostics the flags and the report need, straight off the flows block
    for c in ("_basis_pct", "_pcr_oi", "_oi_chg_5d", "_oi_percentile",
              "_fii_chg", "_dii_chg", "_pledge_chg"):
        if c in raw_blocks["flows"].columns:
            ranked[c] = raw_blocks["flows"][c].reindex(ranked.index)

    regime = risk.assess_regime(prep.benchmark, snapshot, cfg,
                                vix=prep.vix, flows=prep.market_flows)

    picks = risk.apply_portfolio_caps(ranked, prep.sectors, cfg)

    plans, whys, flags = [], [], []
    for sym, row in picks.iterrows():
        plans.append(risk.trade_plan(row, cfg, capital=capital,
                                     exposure=regime.exposure))
        whys.append(scoring.humanise(scoring.explain(row, cfg, top_k=5)))
        flags.append(risk.red_flags(row))
    plan_df = pd.DataFrame(plans, index=picks.index)
    picks = picks.drop(columns=[c for c in plan_df.columns if c in picks.columns])
    picks = picks.join(plan_df)
    picks["why"] = whys
    picks["flags"] = flags

    return RunResult(as_of=prep.as_of, picks=picks, ranked=ranked, regime=regime,
                     universe_size=len(prep.universe_symbols), rejected=prep.rejected,
                     sectors=prep.sectors, snapshot=snapshot, config=cfg,
                     ca_events=prep.ca_events, market_flows=prep.market_flows)


def run(cfg: dict, store: Store, as_of: Optional[pd.Timestamp] = None,
        capital: float = 100000.0, verbose: bool = True) -> RunResult:
    """prepare() + score_from_prepared(), plus persisting the day's ranking."""
    prep = prepare(cfg, store, as_of=as_of, verbose=verbose)
    result = score_from_prepared(prep, cfg, capital=capital)
    store.save_scores(result.as_of.strftime("%Y-%m-%d"),
                      result.ranked.reset_index().rename(columns={"index": "symbol"}))
    return result


def _proxy_benchmark(prices: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Equal-weighted index of the universe — used when NSE index files are missing."""
    p = prices[prices["symbol"].isin(symbols)].copy()
    p["date"] = pd.to_datetime(p["date"])
    piv = p.pivot_table(index="date", columns="symbol", values="close")
    rets = piv.pct_change().mean(axis=1)
    close = (1 + rets.fillna(0)).cumprod() * 1000
    return pd.DataFrame({"date": close.index, "close": close.values})


def _latest_vix(store: Store) -> Optional[float]:
    for name in ("INDIA VIX", "INDIAVIX", "India VIX"):
        df = store.load_index(name)
        if not df.empty:
            return float(df.sort_values("date")["close"].iloc[-1])
    return None
