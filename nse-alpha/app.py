#!/usr/bin/env python3
"""
app.py — the Streamlit UI.

    streamlit run app.py

WHY THIS EXISTS ALONGSIDE THE STATIC SITE
-----------------------------------------
The generated `site/index.html` is a *report*: a fixed record of what the model
said on a given evening, which you can email, archive, and look back at. This app
is a *workbench*: you can move the factor weights and watch the whole market
re-rank underneath you, drill into any stock's chart, and run a backtest with
different parameters without editing config.yaml. They serve different jobs, and
the cron job keeps producing the report either way.

THE THING THAT MAKES IT FEEL FAST
---------------------------------
Scoring is split in two (see pipeline.py). Everything expensive — loading prices,
adjusting for corporate actions, computing ~25 indicators for every stock, running
the six factor blocks — is weight-INDEPENDENT, so it's computed once and cached.
Only the cheap half (winsorise, z-score, blend, rank, size) re-runs when you move
a slider, and that takes about 50 milliseconds across the whole universe. Without
that split, every slider nudge would trigger a two-minute recompute and the app
would be unusable.
"""

from __future__ import annotations

import copy
import os
import sys
from datetime import date, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nsealpha import charts, risk, scoring
from nsealpha.pipeline import (PreparedRun, load_config, prepare,
                               score_from_prepared)
from nsealpha.report import BLOCK_LABELS
from nsealpha.store import Store

st.set_page_config(page_title="NSE Alpha", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

CFG_PATH = "config.yaml"


# ============================================================ cached resources

@st.cache_resource(show_spinner=False)
def get_store(db_path: str) -> Store:
    return Store(db_path)


@st.cache_data(show_spinner=False)
def base_config() -> dict:
    return load_config(CFG_PATH)


@st.cache_data(show_spinner="Loading market data, adjusting for corporate actions, "
                            "and computing factors… (about a minute, once)",
               ttl=60 * 60 * 6, max_entries=3)
def get_prepared(db_path: str, as_of_str: Optional[str], _cfg_fingerprint: str
                 ) -> PreparedRun:
    """
    The expensive half. Cached on (database, as-of date, and the parts of the
    config that actually change the computation — filters and universe, NOT
    weights). Move a weight slider and this does not re-run.
    """
    cfg = load_config(CFG_PATH)
    store = get_store(db_path)
    as_of = pd.Timestamp(as_of_str) if as_of_str else None
    return prepare(cfg, store, as_of=as_of, verbose=False)


@st.cache_data(show_spinner=False, max_entries=400)
def cached_sparkline(symbol: str, as_of_str: str, closes: tuple, dates: tuple,
                     dark: bool):
    """
    A sparkline depends only on the price history, not on the factor weights, so
    it can be cached across re-scores. Without this, every slider nudge redraws
    ten Plotly figures that haven't changed — the single biggest chunk of the
    re-render cost on the picks tab.
    """
    f = pd.DataFrame({"close": list(closes)}, index=pd.to_datetime(list(dates)))
    return charts.sparkline(f, dark=dark, lookback=len(closes))


def spark_for(prep, sym: str, dark: bool, n: int = 120):
    f = prep.frames.get(sym)
    if f is None or len(f) < 30:
        return None, None, None
    tail = f["close"].tail(n)
    fig = cached_sparkline(sym, str(prep.as_of.date()),
                           tuple(round(float(v), 2) for v in tail),
                           tuple(d.isoformat() for d in tail.index), dark)
    return fig, float(tail.min()), float(tail.max())


def cfg_fingerprint(cfg: dict) -> str:
    """Only things that invalidate the expensive half belong in here."""
    f = cfg["filters"]
    u = cfg["universe"]
    d = cfg["data"]
    return "|".join(str(x) for x in [
        u["source"], sorted(u["series_allowed"]),
        f["min_price"], f["max_price"], f["min_avg_turnover_cr"],
        f["min_history_days"], f["min_avg_delivery_pct"], f["max_gap_days"],
        f["exclude_surveillance"], d.get("adjust_corporate_actions", True),
        d["history_days"],
    ])


# ==================================================================== helpers

def is_dark() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def fmt_inr(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"₹{v:,.{nd}f}"


def fmt(v, nd=2, suffix=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{nd}f}{suffix}"


def sidebar_weights(cfg: dict) -> dict:
    """
    Factor weights as sliders. Returns a modified copy of the config — the file on
    disk is never touched, so the cron job keeps using your saved settings.
    """
    st.sidebar.subheader("Factor weights")
    st.sidebar.caption("Move these and the whole market re-ranks instantly. "
                       "Weights are normalised, so they don't need to sum to 1.")

    working = copy.deepcopy(cfg)
    cols = st.sidebar.columns(2)
    new_w = {}
    for i, (block, label) in enumerate(BLOCK_LABELS.items()):
        with cols[i % 2]:
            new_w[block] = st.slider(
                label, 0.0, 0.5, float(cfg["weights"][block]), 0.01,
                key=f"w_{block}",
                help=BLOCK_HELP.get(block, ""))
    total = sum(new_w.values())
    if total <= 0:
        st.sidebar.error("At least one weight must be above zero.")
        return cfg
    working["weights"] = new_w
    st.sidebar.caption(f"Effective split: " + " · ".join(
        f"{BLOCK_LABELS[b]} {100*w/total:.0f}%" for b, w in new_w.items() if w > 0))

    with st.sidebar.expander("Scoring options"):
        working["scoring"]["sector_neutral"] = st.checkbox(
            "Sector-neutral scoring", value=cfg["scoring"]["sector_neutral"],
            help="Z-score within each sector instead of across the whole market. "
                 "Stops the list becoming ten PSU banks in a month when PSU banks "
                 "are ripping — you'd be buying one bet ten times.")
        working["scoring"]["top_n"] = st.slider(
            "How many picks", 3, 30, int(cfg["scoring"]["top_n"]))
        working["scoring"]["min_factor_coverage"] = st.slider(
            "Minimum factor coverage", 0.0, 1.0,
            float(cfg["scoring"]["min_factor_coverage"]), 0.05,
            help="A stock needs data for at least this share of the weighted "
                 "factors to be ranked at all, rather than ranked on fragments.")
        working["risk"]["max_per_sector"] = st.slider(
            "Max picks per sector", 1, 10, int(cfg["risk"]["max_per_sector"]))

    with st.sidebar.expander("Position sizing"):
        working["risk"]["account_risk_pct"] = st.slider(
            "Risk per trade (% of capital)", 0.25, 3.0,
            float(cfg["risk"]["account_risk_pct"]), 0.25)
        working["risk"]["atr_stop_multiple"] = st.slider(
            "Stop distance (× ATR)", 1.0, 4.0,
            float(cfg["risk"]["atr_stop_multiple"]), 0.5)
        working["risk"]["max_stop_pct"] = st.slider(
            "Maximum stop distance (%)", 4.0, 25.0,
            float(cfg["risk"].get("max_stop_pct", 12.0)), 1.0)

    return working


BLOCK_HELP = {
    "technical": "Moving-average structure, ADX, RSI zone, MACD, Bollinger "
                 "squeeze, Supertrend, volume confirmation, and delivery-% trend.",
    "momentum": "1/3/6-month returns, 12-1 momentum, Mansfield relative strength "
                "vs the benchmark, and how consistent the gains have been.",
    "fundamental": "P/E vs the stock's own sector P/E, P/B, PEG, EV/EBITDA, ROE, "
                   "ROCE, margins, debt/equity, growth, promoter holding and pledge.",
    "sentiment": "NSE corporate filings keyword-scored with a 10-day half-life, "
                 "bulk/block deal direction, and reaction to the last results.",
    "trend": "What happened the previous times this stock's setup looked like "
             "today, its Hurst exponent, seasonality and drawdown recovery.",
    "flows": "Futures OI build-up, put-call ratio, futures basis, and quarterly "
             "FII/DII/promoter holding changes. Only ~200 stocks have derivatives.",
}


REGIME_TONES = {
    # Softened, color-coded card — still reads RISK-ON/CAUTION/RISK-OFF at a
    # glance by hue, but without the saturated st.error()-style "something is
    # broken" look. Background is a low-opacity tint of the accent, not a
    # solid fill, so it stays legible in both themes. Text deliberately uses
    # color:inherit rather than a light/dark switch keyed off is_dark() —
    # st.get_option("theme.base") does not reliably reflect the browser's
    # actual rendered theme, and Streamlit's own ambient text color is
    # already correct for whichever theme is really active.
    "RISK-ON": {"accent": "#0ca30c", "bg": "rgba(12,163,12,0.10)"},
    "CAUTION": {"accent": "#c98a12", "bg": "rgba(201,138,18,0.12)"},
    "RISK-OFF": {"accent": "#c9524f", "bg": "rgba(201,82,79,0.12)"},
}


def regime_banner(result):
    R = result.regime
    tone = REGIME_TONES[R.verdict]
    notes_html = "".join(f"<li style='margin:3px 0'>{n}</li>" for n in R.notes)
    st.markdown(
        f"""<div style="background:{tone['bg']};border-left:4px solid {tone['accent']};
                    border-radius:8px;padding:14px 20px;margin-bottom:10px;
                    color:inherit;">
              <div style="font-weight:600;font-size:15px;color:inherit;
                          margin-bottom:6px;">
                Market regime: {R.verdict} — suggested exposure
                {R.exposure*100:.0f}% of your normal size
              </div>
              <ul style="margin:0;padding-left:18px;color:inherit;
                         font-size:14px;line-height:1.55;">
                {notes_html}
              </ul>
            </div>""",
        unsafe_allow_html=True)


# ================================================================ page: picks

def page_picks(result, prep, cfg, capital: float, dark: bool):
    regime_banner(result)

    R = result.regime
    cols = st.columns(5)
    cols[0].metric("Benchmark vs 200-DMA",
                   "—" if R.benchmark_above_200 is None
                   else ("Above" if R.benchmark_above_200 else "Below"),
                   help="Most large drawdowns happen below the 200-day average.")
    cols[1].metric("Market breadth", fmt(R.breadth_pct, 0, "%"),
                   help="Share of the universe trading above its own 50-DMA.")
    cols[2].metric("India VIX", fmt(R.vix, 1))
    if R.fii_20d_cr is not None:
        cols[3].metric("FII net (20 sessions)", f"₹{R.fii_20d_cr:,.0f} cr",
                       delta=None if R.dii_20d_cr is None
                       else f"DII ₹{R.dii_20d_cr:,.0f} cr", delta_color="off")
    else:
        cols[3].metric("Benchmark 1-month", fmt(R.bench_ret_1m, 1, "%"))
    cols[4].metric("Stocks scored", f"{len(result.ranked):,}",
                   help=f"{result.universe_size} passed the tradability filters "
                        f"out of {result.universe_size + len(result.rejected)} screened.")

    st.caption(f"Ranked as of **{result.as_of:%d %b %Y}** · sized for a "
               f"₹{capital:,.0f} account risking "
               f"{cfg['risk']['account_risk_pct']}% per trade, scaled to the regime.")

    st.divider()

    for sym, r in result.picks.iterrows():
        with st.container(border=True):
            head, spark = st.columns([2.2, 1])
            with head:
                st.markdown(f"### {int(r['final_rank'])}. {sym}")
                st.caption(f"{r.get('sector','Unclassified')} · {fmt_inr(r.get('close'))} · "
                           f"composite {r['composite']:+.2f} · "
                           f"ranked {int(r['rank'])} of {len(result.ranked)}")
                chips = st.columns(5)
                for c, (label, val) in zip(chips, [
                        ("1M", fmt(r.get("ret_1m"), 1, "%")),
                        ("3M", fmt(r.get("ret_3m"), 1, "%")),
                        ("RSI", fmt(r.get("rsi14"), 0)),
                        ("ADX", fmt(r.get("adx"), 0)),
                        ("₹cr/day", fmt(r.get("turnover_20d_cr"), 0))]):
                    c.metric(label, val)
            with spark:
                fig, lo, hi = spark_for(prep, sym, dark)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"spark_{sym}",
                                    config={"displayModeBar": False})
                    st.caption(f"120 sessions · low {fmt_inr(lo)} · high {fmt_inr(hi)}")

            left, right = st.columns([1.15, 1])
            with left:
                st.plotly_chart(
                    charts.factor_bars({b: r.get(b) for b in BLOCK_LABELS},
                                       BLOCK_LABELS, cfg["weights"], dark=dark),
                    use_container_width=True, key=f"fb_{sym}",
                    config={"displayModeBar": False})
                for w in (r.get("why") or [])[:5]:
                    st.markdown(f"- {w}")
                flags = r.get("flags") or []
                if flags:
                    st.warning("**Check before you buy:**\n\n"
                               + "\n".join(f"- {f}" for f in flags))
            with right:
                plan = st.columns(2)
                plan[0].markdown(
                    f"**Entry zone**  \n{fmt_inr(r.get('entry_zone_low'))} – "
                    f"{fmt_inr(r.get('entry_zone_high'))}\n\n"
                    f"**Stop loss**  \n{fmt_inr(r.get('stop'))} "
                    f"({fmt(r.get('stop_pct'),1,'%')})\n\n"
                    f"**Targets**  \n{fmt_inr(r.get('target1'))} / "
                    f"{fmt_inr(r.get('target2'))}")
                plan[1].markdown(
                    f"**Quantity**  \n{int(r.get('shares') or 0)} shares\n\n"
                    f"**Position**  \n{fmt_inr(r.get('position_value'), 0)}\n\n"
                    f"**At risk**  \n{fmt_inr(r.get('capital_at_risk'), 0)}")

                with st.expander("Fundamentals, positioning & history"):
                    a, b = st.columns(2)
                    fcf_pos = r.get("_fcf_positive_years")
                    fcf_yrs = r.get("_fcf_years")
                    fcf_txt = ("—" if not (fcf_pos and np.isfinite(fcf_pos))
                               else f"{int(fcf_pos)} of {int(fcf_yrs)}")
                    a.markdown(
                        f"P/E **{fmt(r.get('_pe'),1)}** (sector {fmt(r.get('_sector_pe'),1)})  \n"
                        f"ROE / ROCE **{fmt(r.get('_roe'),1,'%')} / {fmt(r.get('_roce'),1,'%')}**  \n"
                        f"D/E **{fmt(r.get('_de'),2)}**  \n"
                        f"Interest cover **{fmt(r.get('_icr'),1,'×')}**  \n"
                        f"Cash conversion **{fmt(r.get('_cash_conversion'),2,'×')}**  \n"
                        f"FCF positive **{fcf_txt}** years  \n"
                        f"Gross margin **{fmt(r.get('_gross_margin'),1,'%')}**")
                    b.markdown(
                        f"Futures OI 5d **{fmt(r.get('_oi_chg_5d'),1,'%')}**  \n"
                        f"Put-call ratio **{fmt(r.get('_pcr_oi'),2)}**  \n"
                        f"Basis vs spot **{fmt(r.get('_basis_pct'),2,'%')}**  \n"
                        f"FII change QoQ **{fmt(r.get('_fii_chg'),2)}**  \n"
                        f"Edge: **{fmt(r.get('_edge_ret'),2,'%')}** avg over "
                        f"{int(r.get('_edge_n') or 0)} similar setups  \n"
                        f"Hurst **{fmt(r.get('_hurst'),2)}**")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.score_distribution(
            result.ranked, list(result.picks.index), dark=dark),
            use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.plotly_chart(charts.sector_strength(
            result.ranked, result.sectors, dark=dark),
            use_container_width=True, config={"displayModeBar": False})


# =============================================================== page: explore

def page_explore(result, prep, cfg, dark: bool):
    st.subheader("Explore the full ranking")
    st.caption("Every stock that survived the filters and had enough factor "
               "coverage to score. Sort by any column; filter with the controls.")

    ranked = result.ranked.copy()
    ranked["sector"] = [result.sectors.get(s, "Unclassified") for s in ranked.index]

    f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1])
    query = f1.text_input("Search symbol", placeholder="e.g. RELI")
    sectors_available = sorted(ranked["sector"].dropna().unique())
    chosen = f2.multiselect("Sector", sectors_available, default=[])
    min_turnover = f3.number_input("Min ₹cr/day", 0.0, 10000.0,
                                   float(cfg["filters"]["min_avg_turnover_cr"]), 1.0)
    max_rsi = f4.slider("Max RSI", 0, 100, 100,
                        help="Filter out overbought names — set to 70 to exclude "
                             "anything already stretched.")

    view = ranked
    if query:
        view = view[view.index.str.contains(query.upper(), na=False)]
    if chosen:
        view = view[view["sector"].isin(chosen)]
    if "turnover_20d_cr" in view.columns:
        view = view[view["turnover_20d_cr"].fillna(0) >= min_turnover]
    if "rsi14" in view.columns and max_rsi < 100:
        view = view[view["rsi14"].fillna(0) <= max_rsi]

    st.caption(f"{len(view):,} of {len(ranked):,} stocks match.")

    cols = ["rank", "sector", "composite"] + list(BLOCK_LABELS) + [
        "close", "ret_1m", "ret_3m", "ret_6m", "rsi14", "adx",
        "turnover_20d_cr", "pct_from_52w_high", "deliv_ma20", "coverage"]
    cols = [c for c in cols if c in view.columns]
    table = view[cols].copy()
    table.index.name = "Symbol"

    st.dataframe(
        table, use_container_width=True, height=520,
        column_config={
            "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
            "sector": st.column_config.TextColumn("Sector", width="medium"),
            "composite": st.column_config.ProgressColumn(
                "Score", format="%.2f",
                min_value=float(table["composite"].min() if len(table) else -1),
                max_value=float(table["composite"].max() if len(table) else 1)),
            **{b: st.column_config.NumberColumn(BLOCK_LABELS[b], format="%.2f")
               for b in BLOCK_LABELS if b in table.columns},
            "close": st.column_config.NumberColumn("Price", format="₹%.2f"),
            "ret_1m": st.column_config.NumberColumn("1M", format="%.1f%%"),
            "ret_3m": st.column_config.NumberColumn("3M", format="%.1f%%"),
            "ret_6m": st.column_config.NumberColumn("6M", format="%.1f%%"),
            "rsi14": st.column_config.NumberColumn("RSI", format="%.0f"),
            "adx": st.column_config.NumberColumn("ADX", format="%.0f"),
            "turnover_20d_cr": st.column_config.NumberColumn("₹cr/day", format="%.1f"),
            "pct_from_52w_high": st.column_config.NumberColumn("vs 52wH", format="%.1f%%"),
            "deliv_ma20": st.column_config.NumberColumn("Deliv %", format="%.1f"),
            "coverage": st.column_config.NumberColumn("Coverage", format="%.0%"),
        })

    st.download_button("Download this view as CSV",
                       table.to_csv().encode("utf-8"),
                       file_name=f"nse-alpha-{result.as_of:%Y-%m-%d}.csv",
                       mime="text/csv")


# ============================================================ page: deep dive

def page_stock(result, prep, cfg, capital: float, dark: bool):
    st.subheader("Stock deep dive")

    universe = list(result.ranked.index)
    default = list(result.picks.index)[0] if len(result.picks) else universe[0]
    c1, c2, c3 = st.columns([1.4, 1, 2])
    sym = c1.selectbox("Symbol", universe,
                       index=universe.index(default) if default in universe else 0)
    lookback = c2.selectbox("History", [90, 180, 250, 500, 750], index=2,
                            format_func=lambda n: f"{n} sessions")
    panels = c3.multiselect("Panels", ["Volume", "Delivery %", "RSI", "MACD"],
                            default=["Volume", "Delivery %", "RSI"])

    if sym not in prep.frames:
        st.warning("No indicator history for this symbol.")
        return

    row = result.ranked.loc[sym]
    in_picks = sym in result.picks.index
    plan = risk.trade_plan(row, cfg, capital=capital,
                           exposure=result.regime.exposure)

    m = st.columns(6)
    m[0].metric("Rank", f"#{int(row['rank'])}", help=f"of {len(result.ranked)} scored")
    m[1].metric("Composite", f"{row['composite']:+.2f}")
    m[2].metric("Price", fmt_inr(row.get("close")))
    m[3].metric("1M return", fmt(row.get("ret_1m"), 1, "%"))
    m[4].metric("From 52w high", fmt(row.get("pct_from_52w_high"), 1, "%"))
    m[5].metric("ATR", fmt(row.get("atr_pct"), 1, "%"),
                help="Average daily range as a percentage of price — the stock's "
                     "normal travel, and what the stop is sized against.")

    st.plotly_chart(
        charts.price_chart(
            prep.frames[sym], sym, dark=dark, lookback=lookback,
            show_volume="Volume" in panels,
            show_delivery="Delivery %" in panels,
            show_rsi="RSI" in panels,
            show_macd="MACD" in panels,
            trade_levels=plan if plan else None),
        use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Why it ranks where it does")
        st.plotly_chart(
            charts.factor_bars({b: row.get(b) for b in BLOCK_LABELS},
                               BLOCK_LABELS, cfg["weights"], dark=dark),
            use_container_width=True, config={"displayModeBar": False})
        contribs = scoring.explain(row, cfg, top_k=8)
        for label, val in zip(scoring.humanise(contribs), [c[1] for c in contribs]):
            st.markdown(f"- {label}  \n  <span style='color:#898781;font-size:12px'>"
                        f"contribution {val:+.3f}</span>", unsafe_allow_html=True)

    with right:
        st.markdown("#### Trade plan"
                    + ("" if in_picks else "  \n*(not in today's top picks — "
                                           "shown for reference)*"))
        if plan:
            p1, p2 = st.columns(2)
            p1.metric("Entry zone", f"{fmt_inr(plan['entry_zone_low'])} – "
                                    f"{fmt_inr(plan['entry_zone_high'])}")
            p1.metric("Stop loss", fmt_inr(plan["stop"]),
                      delta=f"{plan['stop_pct']:.1f}%", delta_color="inverse")
            p1.metric("Quantity", f"{plan['shares']} shares")
            p2.metric("Target 1", fmt_inr(plan["target1"]),
                      delta=f"+{plan['target1_pct']:.1f}%")
            p2.metric("Target 2", fmt_inr(plan["target2"]),
                      delta=f"+{plan['target2_pct']:.1f}%")
            p2.metric("Capital at risk", fmt_inr(plan["capital_at_risk"], 0))
        else:
            st.info("Not enough data to size a position in this name.")

        flags = risk.red_flags(row)
        if flags:
            st.warning("**Red flags**\n\n" + "\n".join(f"- {f}" for f in flags))

        st.markdown("#### Sub-factor detail")
        sub = {}
        for b in BLOCK_LABELS:
            for c in result.ranked.columns:
                if c.startswith(f"{b}__"):
                    sub[c.replace(f"{b}__", f"{BLOCK_LABELS[b]}: ")] = row[c]
        sub_df = pd.DataFrame({"z-score": pd.Series(sub)}).dropna()
        st.dataframe(sub_df.style.format("{:+.2f}"), use_container_width=True,
                     height=300)


# ============================================================== page: backtest

def page_backtest(cfg, db_path: str, dark: bool):
    st.subheader("Walk-forward backtest")
    st.info("Each period re-runs the **entire model** with `as_of` set to that "
            "date, so nothing sees the future. That also means it is slow — "
            "roughly 15–25 seconds per period. Start with 10.", icon="⏱️")

    c1, c2, c3, c4 = st.columns(4)
    start = c1.date_input("Start from", value=pd.Timestamp(cfg["backtest"]["start"]))
    hold = c2.slider("Hold period (trading days)", 1, 30,
                     int(cfg["backtest"]["rebalance_days"]))
    periods = c3.slider("Periods to test", 3, 60, 10)
    cost = c4.slider("Round-trip cost (bps)", 0, 150,
                     int(cfg["backtest"]["cost_bps"]),
                     help="Brokerage + STT + stamp duty + your realistic slippage. "
                          "35bps is optimistic for anything but large caps.")

    st.warning(
        "**Read the result sceptically.** Fundamentals are stored as a current "
        "snapshot rather than point-in-time, so the fundamental block carries "
        "look-ahead bias here. For a clean read, set that weight to 0 in the "
        "sidebar, re-run, and compare the two numbers. If the backtest shows a "
        "spectacular Sharpe, assume a bug before assuming an edge.", icon="⚠️")

    if st.button("Run backtest", type="primary"):
        from nsealpha.backtest import run_backtest
        bt_cfg = copy.deepcopy(cfg)
        bt_cfg["backtest"].update({"start": str(start), "rebalance_days": hold,
                                   "cost_bps": cost,
                                   "top_n": cfg["scoring"]["top_n"]})
        store = get_store(db_path)
        prog = st.progress(0.0, text="Starting…")

        def on_progress(done: int, total: int, label: str):
            prog.progress(min(done / max(total, 1), 1.0),
                          text=f"Period {done + 1} of {total} — {label}")

        try:
            res = run_backtest(bt_cfg, store, max_periods=periods,
                               verbose=False, progress=on_progress)
            prog.empty()
            st.session_state["bt"] = res
        except Exception as e:
            prog.empty()
            st.error(f"Backtest failed: {e}")

    res = st.session_state.get("bt")
    if res is None:
        return

    s = res.stats
    if not s.get("reliable", True):
        st.error(
            f"**Only {s['periods']} periods — annualised return and Sharpe are "
            f"withheld.** Annualising a handful of {s['hold_days']}-day windows "
            f"turns three coin flips into a headline CAGR. Re-run with at least "
            f"{s['min_reliable_periods']} periods before reading anything into "
            f"this.", icon="🚫")

    m = st.columns(6)
    m[0].metric("Total return", f"{s['total_return_pct']:.1f}%",
                delta=f"{s['total_return_pct'] - s['benchmark_return_pct']:+.1f}% vs index")
    if s.get("reliable", True):
        m[1].metric("Annualised", f"{s['cagr_pct']:.1f}%")
        m[2].metric("Sharpe", f"{s['sharpe']:.2f}")
    else:
        m[1].metric("Annualised", "—", help="Too few periods to annualise.")
        m[2].metric("Sharpe", "—", help="Too few periods for a meaningful Sharpe.")
    m[3].metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")
    m[4].metric("Periods positive", f"{s['period_win_rate_pct']:.0f}%")
    m[5].metric("Beat benchmark", f"{s['beat_benchmark_pct']:.0f}%")
    st.caption(f"{s['periods']} rebalances · {s['hold_days']}-day hold · "
               f"{s['trades']} positions taken")

    st.plotly_chart(charts.equity_curve(res.periods, dark=dark),
                    use_container_width=True)
    st.plotly_chart(charts.period_returns(res.periods, dark=dark),
                    use_container_width=True)

    with st.expander("Every period"):
        st.dataframe(res.periods, use_container_width=True)
    with st.expander("Every trade"):
        st.dataframe(res.trades, use_container_width=True)
        st.download_button("Download trades CSV",
                           res.trades.to_csv(index=False).encode("utf-8"),
                           file_name="backtest_trades.csv", mime="text/csv")


# ======================================================= page: email & schedule

def page_email(result, cfg, db_path: str):
    from nsealpha import mailer, scheduler
    from nsealpha.pipeline import refresh_data

    project_dir = Path(__file__).resolve().parent
    mailer.load_env(project_dir / ".env")
    mcfg = mailer.mail_config(project_dir / ".env")

    # ------------------------------------------------------------ data refresh
    st.subheader("Data refresh")
    st.caption("Pull today's bhavcopy, indices, corporate actions, F&O and "
               "fundamentals from NSE. The app never fetches on its own — this "
               "only ever runs when you click the button below.")

    store = get_store(db_path)
    r1, r2 = st.columns([2, 1])
    r1.caption(f"Latest session on file: **{store.latest_price_date()}**")
    skip_fund = r2.checkbox(
        "Skip fundamentals (faster)", value=False,
        help="Skips the slow per-symbol ratios/statements pass. Prices, "
             "indicators and F&O still refresh.")

    if st.button("Refresh today's data now", type="primary",
                 use_container_width=True):
        try:
            with st.spinner("Fetching from NSE — a same-day refresh usually "
                            "takes 2-5 minutes..."):
                refresh_data(cfg, store, full_backfill=False,
                             skip_fundamentals=skip_fund, verbose=False)
            st.cache_data.clear()
            st.success(f"Refreshed. Latest session: {store.latest_price_date()}.")
            st.rerun()
        except Exception as e:
            st.error(f"Refresh failed: {e}")
            if store.latest_price_date() is None:
                st.info("No cached data to fall back on. If NSE is blocking "
                        "this connection, try setting `data.server_mode: true` "
                        "in config.yaml.")

    st.caption("First time ever running this? Use the command line instead — "
               "`python run_daily.py --refresh --backfill` pulls ~900 days of "
               "history and takes 20-60 minutes, too long for a browser tab.")

    st.divider()

    st.subheader("Daily email digest")
    st.caption("The top 10, what changed since yesterday, entry/stop/targets and "
               "red flags — in your inbox each evening.")

    sched = scheduler.status(project_dir)
    c1, c2, c3 = st.columns(3)
    c1.metric("Email configured", "Yes" if mcfg.ok else "No",
              help="SMTP host, user, password and at least one recipient.")
    c2.metric("Daily schedule", "On" if sched.installed else "Off")
    if sched.installed and sched.next_run:
        c3.metric("Next send", sched.next_run.strftime("%a %d %b, %H:%M"))
    elif sched.last_run:
        c3.metric("Last run", sched.last_run,
                  help="Most recent scheduled run found in logs/.")
    else:
        c3.metric("Next send", "—")

    if sched.last_run and sched.last_run_ok is False:
        st.error(f"The last scheduled run at {sched.last_run} did not finish "
                 f"successfully — check `logs/cron.log`. A silent cron failure "
                 f"otherwise looks exactly like a quiet market day.", icon="⚠️")

    st.divider()

    # ------------------------------------------------------------- SMTP setup
    with st.expander("Email settings", expanded=not mcfg.ok):
        st.markdown(
            "**Gmail needs an App Password, not your account password.** "
            "Google Account → Security → turn on 2-Step Verification → App "
            "passwords → Mail → generate. The 16-character code goes below "
            "(spaces are fine). Outlook is `smtp-mail.outlook.com:587`, "
            "Zoho India is `smtp.zoho.in:587`.")

        with st.form("smtp"):
            a, b = st.columns(2)
            host = a.text_input("SMTP host", value=mcfg.host or "smtp.gmail.com")
            port = b.number_input("Port", 1, 65535, int(mcfg.port or 587),
                                  help="587 for STARTTLS, 465 for implicit TLS.")
            user = a.text_input("Username", value=mcfg.user)
            password = b.text_input("Password / app password", type="password",
                                    value=mcfg.password)
            sender = a.text_input("From address", value=mcfg.sender or mcfg.user)
            to = b.text_input("Send to", value=", ".join(mcfg.recipients),
                              help="Comma-separated for more than one recipient.")
            site_url = st.text_input(
                "Public site URL (optional)", value=os.environ.get("SITE_URL", ""),
                help="If you publish site/ somewhere, the email gets a button "
                     "linking to the full dashboard.")

            if st.form_submit_button("Save settings", type="primary"):
                vals = {"SMTP_HOST": host, "SMTP_PORT": str(int(port)),
                        "SMTP_USER": user, "SMTP_PASSWORD": password,
                        "EMAIL_FROM": sender or user, "EMAIL_TO": to}
                if site_url.strip():
                    vals["SITE_URL"] = site_url.strip()
                p = mailer.save_env(vals, project_dir / ".env")
                st.success(f"Saved to `{p.name}` (readable only by you).")
                st.rerun()

        st.caption("Stored in plaintext in `.env`, permissions 0600, and already "
                   "in `.gitignore`. Any mail client needs a credential it can "
                   "read; there's no way around that — just don't commit it.")

    # ------------------------------------------------------------ send now
    st.markdown("#### Send now")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("Send today's digest", disabled=not mcfg.ok,
                     use_container_width=True, type="primary"):
            try:
                mailer.send_digest(get_store(db_path), result,
                                   site_url=os.environ.get("SITE_URL"),
                                   env_file=project_dir / ".env")
                st.success(f"Sent to {', '.join(mcfg.recipients)}.")
            except Exception as e:
                msg = str(e)
                st.error(f"Could not send: {msg}")
                if "Username and Password not accepted" in msg or "5.7.8" in msg:
                    st.info("Gmail rejects normal account passwords over SMTP — "
                            "you need an App Password. See the note above.")
    with s2:
        if st.button("Preview without sending", use_container_width=True):
            path = mailer.send_digest(
                get_store(db_path), result, site_url=os.environ.get("SITE_URL"),
                dry_run=True, out_dir=Path(cfg["data"]["site_dir"]))
            st.session_state["digest_preview"] = str(path)

    preview = st.session_state.get("digest_preview")
    if preview and Path(preview).exists():
        with st.expander("Digest preview", expanded=True):
            st.components.v1.html(Path(preview).read_text(encoding="utf-8"),
                                  height=620, scrolling=True)

    if not mcfg.ok:
        st.info("Fill in the email settings above to enable sending.", icon="✉️")

    st.divider()

    # -------------------------------------------------------------- schedule
    st.markdown("#### Fetch it automatically, every day")

    st.info(
        "**This installs a job in your operating system's scheduler**, not in "
        "this app. Streamlit only runs Python while a browser tab is open, so a "
        "toggle that lived in here would stop working the moment you closed it. "
        f"On this machine that means **{sched.detail}**, and it keeps running "
        "with the app closed, the browser quit and this tab long forgotten. "
        "The job always refreshes data; emailing the digest is an optional "
        "add-on below.",
        icon="🗓️")

    if not sched.available:
        st.warning(f"Automatic scheduling isn't available here. {sched.detail}",
                   icon="⚠️")
        with st.expander("Set it up by hand instead", expanded=True):
            st.markdown("Run `crontab -e` and paste this in:")
            st.code(sched.manual_line, language="bash")
        return

    # Deliberately two selectboxes rather than st.time_input: the time widget
    # renders through the browser's Intl locale data, which is missing or broken
    # in some environments (headless Chromium, trimmed OS locale packs) and takes
    # the whole section down with a RangeError when it is. Hour and minute
    # dropdowns have no locale dependency and read just as clearly.
    d1, d2, d3, d4 = st.columns([0.8, 0.8, 1.1, 1.5])
    default_h = sched.hour if sched.hour is not None else 19
    default_m = sched.minute if sched.minute is not None else 0
    hour = d1.selectbox("Hour", list(range(24)), index=default_h,
                        format_func=lambda h: f"{h:02d}")
    minute = d2.selectbox("Minute", [0, 15, 30, 45],
                          index=[0, 15, 30, 45].index(default_m)
                          if default_m in (0, 15, 30, 45) else 0,
                          format_func=lambda m: f"{m:02d}")
    weekdays = d3.checkbox("Trading days only (Mon–Fri)",
                           value=sched.weekdays_only)
    d4.caption("NSE publishes the full bhavcopy, including delivery data, in the "
               "early evening. Before about 6:30 pm IST you'd be scoring "
               "yesterday's close, so 19:00 is the sensible default.")
    send_time = dt_time(int(hour), int(minute))

    # Whether the currently-installed job already emails is read back from the
    # command it actually runs where that's available (crontab always exposes
    # it; schtasks needs /V and a recent enough scheduler.py, see status()).
    # Falls back to "on if email is configured" for a first-time setup.
    already_emails = ("--email" in sched.command) if sched.command else mcfg.ok
    also_email = st.checkbox(
        "Also email the digest", value=already_emails and mcfg.ok,
        disabled=not mcfg.ok,
        help="Adds --email to the scheduled run. Needs email settings saved "
             "above — leave unchecked to just refresh data every day with no "
             "email sent." if mcfg.ok else
             "Save your email settings above to enable this.")
    extra_flags = "--refresh --email" if also_email else "--refresh"

    b1, b2 = st.columns(2)
    with b1:
        label = "Update schedule" if sched.installed else (
            "Turn on daily refresh + email" if also_email else "Turn on daily refresh")
        if st.button(label, type="primary", use_container_width=True):
            try:
                new = scheduler.install(project_dir, hour=send_time.hour,
                                        minute=send_time.minute,
                                        weekdays_only=weekdays,
                                        extra_flags=extra_flags)
                nxt = (new.next_run.strftime("%A %d %b at %H:%M")
                       if new.next_run else "the next scheduled slot")
                what = "Daily refresh + email" if also_email else "Daily refresh"
                st.success(f"{what} is on. First run: {nxt}.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not install the schedule: {e}")
    with b2:
        if st.button("Turn off", use_container_width=True,
                     disabled=not sched.installed):
            try:
                scheduler.remove(project_dir)
                st.success("Daily schedule turned off. Your other scheduled "
                           "jobs were left untouched.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not remove the schedule: {e}")

    if sched.installed:
        st.success(
            f"Scheduled: **{'Mon–Fri' if sched.weekdays_only else 'every day'} at "
            f"{sched.hour:02d}:{sched.minute:02d}**"
            + (f" · next {sched.next_run:%a %d %b %H:%M}" if sched.next_run else ""),
            icon="✅")
        with st.expander("What exactly was installed"):
            st.code(sched.command or sched.manual_line, language="bash")
            st.caption("Output goes to `logs/cron.log`. The entry is wrapped in "
                       "marker comments and only that block is ever touched, so "
                       "your other cron jobs are safe.")

    st.caption(
        "**One caveat worth knowing:** the job runs on *this machine*. If it is "
        "asleep or switched off at the scheduled time, the scheduler does not "
        "catch up afterwards — you simply get no refresh (and no email, if that's "
        "on) that day. A laptop that's shut by 7pm is a poor host for this; an "
        "always-on desktop or a small VPS is a good one.")


# ============================================================== page: data

def page_data(result, prep, cfg, db_path: str, dark: bool):
    st.subheader("Data quality")
    st.caption("Silent data problems are the main way a screen like this misleads "
               "you, so they get a page rather than a log line nobody reads.")

    ca = prep.ca_events
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corporate actions applied", 0 if ca is None or ca.empty else len(ca))
    c2.metric("Cross-source disagreements",
              0 if ca is None or ca.empty else int(ca["disagrees"].sum()),
              help="The corporate-actions filing and the bhavcopy's implied factor "
                   "differed. The stated filing was used — check these by hand.")
    c3.metric("Stocks with derivatives",
              0 if prep.fno is None or prep.fno.empty
              else len(set(prep.fno.index) & set(prep.snapshot.index)))
    c4.metric("Latest session", f"{result.as_of:%d %b %Y}")

    st.markdown("#### Why history has to be adjusted")
    st.markdown(
        "NSE bhavcopy carries **raw traded prices**. A 1:5 split therefore appears "
        "as a −80% day, which silently corrupts every momentum horizon, the "
        "52-week high, ATR (and so every stop this tool suggests), realised "
        "volatility, OBV, the Hurst exponent and the historical-edge factor. "
        "Adjustments below are derived from the bhavcopy's own `PREV_CLOSE` column "
        "— NSE publishes the *adjusted* previous close on ex-dates, because that "
        "drives the day's circuit limits — and cross-checked against the corporate "
        "actions filing. **Today's prices are never adjusted**, so the entry and "
        "stop levels shown elsewhere are the real numbers on your terminal.")

    if ca is not None and not ca.empty:
        show = ca.sort_values("date", ascending=False).copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%d %b %Y")
        only_bad = st.checkbox("Show only disagreements", value=False)
        if only_bad:
            show = show[show["disagrees"]]
        st.dataframe(show[["symbol", "date", "factor", "source", "implied",
                           "parsed", "disagrees", "purpose"]],
                     use_container_width=True, height=320,
                     column_config={
                         "factor": st.column_config.NumberColumn(format="%.4f"),
                         "implied": st.column_config.NumberColumn(format="%.4f"),
                         "parsed": st.column_config.NumberColumn(format="%.4f"),
                         "purpose": st.column_config.TextColumn(width="large")})
    else:
        st.info("No corporate actions on file for this window.")

    st.divider()
    st.markdown("#### Factor coverage")
    st.caption("Missing is not the same as average. A stock without listed "
               "derivatives is scored as *missing* on the flow sub-factors and the "
               "weights redistribute — treating it as neutral would quietly bias "
               "the ranking toward large caps.")
    cov = pd.DataFrame({
        "Block": [BLOCK_LABELS[b] for b in BLOCK_LABELS if b in result.ranked],
        "Stocks with data (%)": [round(result.ranked[b].notna().mean() * 100)
                                 for b in BLOCK_LABELS if b in result.ranked]})
    st.dataframe(cov, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### What got filtered out, and why")
    if prep.rejected is not None and not prep.rejected.empty:
        summary = (prep.rejected.groupby("reason").size()
                   .sort_values(ascending=False).rename("Stocks").reset_index())
        summary.columns = ["Reason", "Stocks"]
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Nothing was filtered out.")

    st.divider()
    st.markdown("#### Institutional cash flow")
    store = get_store(db_path)
    mf = store.load_market_flows(
        since=(result.as_of - pd.Timedelta(days=60)).strftime("%Y-%m-%d"))
    if not mf.empty and mf["metric"].str.contains("net_cr").any():
        st.plotly_chart(charts.flow_chart(mf, dark=dark), use_container_width=True)
        st.caption("FIIs move the index; DIIs, fed by domestic SIP money, "
                   "increasingly absorb what FIIs sell. The divergence between them "
                   "says more than either number alone.")
    else:
        st.info("No FII/DII data yet. This endpoint has no historical archive — "
                "it accumulates from the day you start running `--refresh` daily.")


# ==================================================================== main

def main():
    cfg0 = base_config()
    db_path = cfg0["data"]["db_path"]
    dark = is_dark()

    st.sidebar.title("NSE Alpha")
    st.sidebar.caption("Multi-factor ranking for Indian equities")

    store = get_store(db_path)
    latest = store.latest_price_date()
    if latest is None:
        st.error("No market data in the database yet.")
        st.markdown(
            "Run one of these in the project folder first:\n\n"
            "```bash\n"
            "python run_daily.py --simulate        # synthetic data, no network\n"
            "python run_daily.py --refresh --backfill   # real NSE data\n"
            "```")
        st.stop()

    if store.get_meta("simulated"):
        st.sidebar.warning("**Simulated data.** This database was built by "
                           "`--simulate`. The plumbing is real; the numbers are "
                           "fiction. Do not trade from it.", icon="🧪")

    as_of = st.sidebar.date_input(
        "Score as of", value=pd.Timestamp(latest).date(),
        max_value=pd.Timestamp(latest).date(),
        help="Roll the model back in time. Nothing after this date is used, so "
             "you can check what it would have said last Tuesday.")
    capital = st.sidebar.number_input("Account size (₹)", 10_000, 100_000_000,
                                      100_000, 10_000,
                                      help="Position sizes are computed from this.")

    cfg = sidebar_weights(cfg0)

    with st.sidebar:
        st.divider()
        if st.button("Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
        st.caption(f"Database: `{db_path}`  \nLatest session: **{latest}**")

    try:
        prep = get_prepared(db_path, str(as_of), cfg_fingerprint(cfg))
    except Exception as e:
        st.error(f"Could not prepare the data: {e}")
        st.stop()

    result = score_from_prepared(prep, cfg, capital=capital)
    result.simulated = bool(store.get_meta("simulated"))

    st.title("NSE Alpha")
    if result.simulated:
        st.caption("🧪 Running on simulated data.")

    tabs = st.tabs([f"Top {len(result.picks)}", "Explore", "Stock deep dive",
                    "Backtest", "Email & schedule", "Data quality"])
    with tabs[0]:
        page_picks(result, prep, cfg, capital, dark)
    with tabs[1]:
        page_explore(result, prep, cfg, dark)
    with tabs[2]:
        page_stock(result, prep, cfg, capital, dark)
    with tabs[3]:
        page_backtest(cfg, db_path, dark)
    with tabs[4]:
        page_email(result, cfg, db_path)
    with tabs[5]:
        page_data(result, prep, cfg, db_path, dark)

    st.divider()
    st.caption(
        "**This is not investment advice.** It is a quantitative screen — a way of "
        "narrowing two thousand stocks to ten worth researching properly. The model "
        "cannot see pending litigation, related-party transactions, accounting "
        "irregularities or promoter behaviour. Every factor is derived from past "
        "data, and the relationship between past and future returns is unstable. "
        "Position sizes and stops are arithmetic illustrations of a fixed-risk rule, "
        "not recommendations. Do your own research, consider consulting a "
        "SEBI-registered investment adviser, and never risk money you cannot afford "
        "to lose.")


if __name__ == "__main__":
    main()
