"""
charts.py — Plotly figures for the Streamlit app.

Design rules followed here, so they don't have to be re-argued per chart:

  * ONE AXIS PER PANEL. Never a dual-axis chart. Price, volume, delivery %, RSI
    and MACD are different measures on different scales, so they get their own
    stacked subplot rows sharing an x-axis — not a second y-axis crammed onto the
    price panel. (Dual-axis charts let you imply any correlation you like by
    choosing the scales, which is exactly why they're banned.)
  * Colour follows the ENTITY, not its rank. SMA20 is always slot 2, SMA50 always
    slot 3, SMA200 always slot 4, whatever else is on screen.
  * Diverging encoding for polarity. Factor scores are "above or below the
    universe average", so they use the blue/red diverging pair with a neutral
    midpoint — never a sequential ramp, which would imply magnitude only.
  * Legend whenever two or more series share a panel; a single-series panel is
    named by its title instead.
  * Recessive grid and axes; the data is the loudest thing on screen.
  * Every panel carries a hover layer, with a unified crosshair on time series.

Palette values are the validated defaults (see the dataviz reference). Both light
and dark are explicitly chosen sets, not one flipped into the other.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------------- palette

LIGHT = {
    "surface": "#fcfcfb", "plane": "#f9f9f7",
    "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100",
    "s5": "#e87ba4", "s6": "#008300", "s7": "#4a3aa7", "s8": "#e34948",
    "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec",
    "up": "#0ca30c", "down": "#d03b3b", "warn": "#fab219",
}

DARK = {
    "surface": "#1a1a19", "plane": "#0d0d0d",
    "text": "#ffffff", "text2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835",
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "s4": "#c98500",
    "s5": "#d55181", "s6": "#008300", "s7": "#9085e9", "s8": "#e66767",
    "pos": "#3987e5", "neg": "#e66767", "mid": "#383835",
    "up": "#0ca30c", "down": "#d03b3b", "warn": "#fab219",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(dark: bool = False) -> Dict[str, str]:
    return DARK if dark else LIGHT


def _base_layout(fig: go.Figure, P: Dict[str, str], height: int,
                 title: Optional[str] = None, legend: bool = False,
                 margin_top: int = 30) -> go.Figure:
    # Set the title only when there is one. Passing an empty/None title through
    # update_layout leaves a stray placeholder annotation in the rendered figure.
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color=P["text"]),
                                     x=0, xanchor="left"))
    fig.update_layout(
        height=height,
        paper_bgcolor=P["surface"], plot_bgcolor=P["surface"],
        font=dict(family=FONT, size=12, color=P["text2"]),
        margin=dict(l=8, r=8, t=margin_top if title else 12, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["axis"],
                        font=dict(family=FONT, size=12, color=P["text"])),
        dragmode="pan",
    )
    fig.update_xaxes(showgrid=False, linecolor=P["axis"], zeroline=False,
                     tickfont=dict(color=P["muted"], size=11),
                     showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor=P["axis"])
    fig.update_yaxes(gridcolor=P["grid"], linecolor=P["axis"], zeroline=False,
                     tickfont=dict(color=P["muted"], size=11))
    return fig


# -------------------------------------------------------------------- spark

def sparkline(f: pd.DataFrame, *, dark: bool = False, lookback: int = 120,
              height: int = 130) -> go.Figure:
    """
    A thumbnail for a pick card. One series, so no legend — the card title names
    the stock. No axes, no grid, no toolbar: at this size any chrome is noise, and
    the exact numbers live in the card beside it.
    """
    P = palette(dark)
    d = f.tail(lookback)
    c = d["close"]
    if len(c) < 5:
        return go.Figure()
    rising = float(c.iloc[-1]) >= float(c.iloc[0])
    colour = P["pos"] if rising else P["neg"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=c, mode="lines", name="Close",
        line=dict(color=colour, width=2, shape="spline", smoothing=0.4),
        fill="tozeroy", fillcolor=_rgba(colour, 0.08),
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[d.index[-1]], y=[c.iloc[-1]], mode="markers", showlegend=False,
        marker=dict(color=colour, size=8, line=dict(color=P["surface"], width=2)),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=height, showlegend=False,
        paper_bgcolor=P["surface"], plot_bgcolor=P["surface"],
        margin=dict(l=0, r=0, t=6, b=0),
        font=dict(family=FONT, size=11, color=P["muted"]),
        hovermode="x", dragmode=False,
        hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["axis"],
                        font=dict(family=FONT, size=12, color=P["text"])),
    )
    pad = (c.max() - c.min()) * 0.12 or 1
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True,
                     range=[c.min() - pad, c.max() + pad])
    return fig


def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------- price & indicators

def price_chart(f: pd.DataFrame, symbol: str, *, dark: bool = False,
                lookback: int = 250, show_volume: bool = True,
                show_delivery: bool = True, show_rsi: bool = True,
                show_macd: bool = False, show_bollinger: bool = True,
                show_supertrend: bool = True,
                trade_levels: Optional[Dict[str, float]] = None) -> go.Figure:
    """
    Stacked panels sharing one time axis. Each panel has exactly one y-scale.
    """
    P = palette(dark)
    d = f.tail(lookback)

    rows: List[str] = ["price"]
    if show_volume:
        rows.append("volume")
    if show_delivery and "deliv_pct" in d.columns and d["deliv_pct"].notna().any():
        rows.append("delivery")
    if show_rsi:
        rows.append("rsi")
    if show_macd:
        rows.append("macd")

    heights = {"price": 0.50, "volume": 0.14, "delivery": 0.13,
               "rsi": 0.14, "macd": 0.16}
    hs = np.array([heights[r] for r in rows], dtype=float)
    hs = hs / hs.sum()

    titles = {"price": None, "volume": "Volume", "delivery": "Delivery %",
              "rsi": "RSI (14)", "macd": "MACD (12,26,9)"}

    sub_titles = [titles[r] or " " for r in rows]
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.035, row_heights=list(hs),
                        subplot_titles=sub_titles)

    r = {name: i + 1 for i, name in enumerate(rows)}

    # ---------------- price panel ----------------
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name=symbol,
        increasing=dict(line=dict(color=P["up"], width=1), fillcolor=P["up"]),
        decreasing=dict(line=dict(color=P["down"], width=1), fillcolor=P["down"]),
        showlegend=False,
    ), row=r["price"], col=1)

    # Moving averages keep fixed colour slots so identity never shifts.
    for col, label, slot in (("sma20", "SMA 20", "s2"),
                             ("sma50", "SMA 50", "s3"),
                             ("sma200", "SMA 200", "s4")):
        if col in d.columns and d[col].notna().any():
            fig.add_trace(go.Scatter(
                x=d.index, y=d[col], name=label, mode="lines",
                line=dict(color=P[slot], width=2),
                hovertemplate=label + ": ₹%{y:,.2f}<extra></extra>",
            ), row=r["price"], col=1)

    if show_bollinger and "bb_upper" in d.columns and d["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_upper"], name="Bollinger",
                                 mode="lines", line=dict(color=P["muted"], width=1,
                                                         dash="dot"),
                                 hoverinfo="skip", showlegend=True),
                      row=r["price"], col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["bb_lower"], name="Bollinger lower",
                                 mode="lines", line=dict(color=P["muted"], width=1,
                                                         dash="dot"),
                                 fill="tonexty", fillcolor="rgba(137,135,129,0.07)",
                                 hoverinfo="skip", showlegend=False),
                      row=r["price"], col=1)

    if show_supertrend and "supertrend" in d.columns and d["supertrend"].notna().any():
        st_up = d["supertrend"].where(d["supertrend_dir"] > 0)
        st_dn = d["supertrend"].where(d["supertrend_dir"] <= 0)
        fig.add_trace(go.Scatter(x=d.index, y=st_up, name="Supertrend (buy)",
                                 mode="lines", line=dict(color=P["s6"], width=2),
                                 connectgaps=False,
                                 hovertemplate="Supertrend: ₹%{y:,.2f}<extra></extra>"),
                      row=r["price"], col=1)
        fig.add_trace(go.Scatter(x=d.index, y=st_dn, name="Supertrend (sell)",
                                 mode="lines", line=dict(color=P["s8"], width=2),
                                 connectgaps=False,
                                 hovertemplate="Supertrend: ₹%{y:,.2f}<extra></extra>"),
                      row=r["price"], col=1)

    # Trade levels are annotations, not series — they're reference lines.
    if trade_levels:
        for key, label, colour, dash in (("stop", "Stop", P["down"], "dash"),
                                         ("entry", "Entry", P["muted"], "dot"),
                                         ("target1", "Target 1", P["up"], "dash"),
                                         ("target2", "Target 2", P["up"], "dot")):
            v = trade_levels.get(key)
            if v is None or not np.isfinite(v):
                continue
            fig.add_hline(y=float(v), line=dict(color=colour, width=1, dash=dash),
                          annotation_text=f"{label} ₹{v:,.0f}",
                          annotation_position="top right",
                          annotation_font=dict(size=10, color=colour),
                          annotation_bgcolor=P["surface"],
                          annotation_yshift=2,
                          row=r["price"], col=1)

    fig.update_yaxes(title_text=None, tickprefix="₹", row=r["price"], col=1)

    # ---------------- volume ----------------
    if "volume" in r:
        up = d["close"] >= d["close"].shift()
        fig.add_trace(go.Bar(
            x=d.index, y=d["volume"], name="Volume",
            marker=dict(color=np.where(up, P["up"], P["down"]),
                        line=dict(width=0)),
            opacity=0.55, showlegend=False,
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ), row=r["volume"], col=1)
        if "vol_surge" in d.columns:
            fig.add_hline(y=float(d["volume"].tail(20).mean()),
                          line=dict(color=P["muted"], width=1, dash="dot"),
                          row=r["volume"], col=1)

    # ---------------- delivery % ----------------
    if "delivery" in r:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["deliv_pct"], name="Delivery %", mode="lines",
            line=dict(color=P["s1"], width=1), opacity=0.45, showlegend=False,
            hovertemplate="Delivery: %{y:.1f}%<extra></extra>",
        ), row=r["delivery"], col=1)
        if "deliv_ma20" in d.columns:
            fig.add_trace(go.Scatter(
                x=d.index, y=d["deliv_ma20"], name="Delivery 20d avg", mode="lines",
                line=dict(color=P["s1"], width=2), showlegend=False,
                hovertemplate="Delivery 20d: %{y:.1f}%<extra></extra>",
            ), row=r["delivery"], col=1)
        fig.update_yaxes(ticksuffix="%", row=r["delivery"], col=1)

    # ---------------- RSI ----------------
    if "rsi" in r and "rsi14" in d.columns:
        fig.add_hrect(y0=55, y1=70, fillcolor=P["s6"], opacity=0.07,
                      line_width=0, row=r["rsi"], col=1)
        for lvl, lbl in ((70, "overbought"), (30, "oversold")):
            fig.add_hline(y=lvl, line=dict(color=P["axis"], width=1, dash="dot"),
                          row=r["rsi"], col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["rsi14"], name="RSI", mode="lines",
            line=dict(color=P["s7"], width=2), showlegend=False,
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=r["rsi"], col=1)
        fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=r["rsi"], col=1)

    # ---------------- MACD ----------------
    if "macd" in r and "macd" in d.columns:
        fig.add_trace(go.Bar(
            x=d.index, y=d["macd_hist"], name="Histogram",
            marker=dict(color=np.where(d["macd_hist"] >= 0, P["pos"], P["neg"]),
                        line=dict(width=0)),
            opacity=0.5, showlegend=False,
            hovertemplate="Histogram: %{y:.2f}<extra></extra>",
        ), row=r["macd"], col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", mode="lines",
                                 line=dict(color=P["s1"], width=2), showlegend=False,
                                 hovertemplate="MACD: %{y:.2f}<extra></extra>"),
                      row=r["macd"], col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd_signal"], name="Signal",
                                 mode="lines", line=dict(color=P["s2"], width=2),
                                 showlegend=False,
                                 hovertemplate="Signal: %{y:.2f}<extra></extra>"),
                      row=r["macd"], col=1)
        fig.add_hline(y=0, line=dict(color=P["axis"], width=1), row=r["macd"], col=1)

    height = 240 + 120 * (len(rows) - 1)
    _base_layout(fig, P, height, legend=True, margin_top=44)
    fig.update_layout(xaxis_rangeslider_visible=False, bargap=0.15)
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = P["muted"]
        ann.xanchor = "left"
        ann.x = 0
    return fig


# ------------------------------------------------------------- factor bars

def factor_bars(scores: Dict[str, float], labels: Dict[str, str],
                weights: Optional[Dict[str, float]] = None,
                *, dark: bool = False, height: int = 210) -> go.Figure:
    """
    Diverging horizontal bars: a block score is polarity (above or below the
    universe average), so blue right / red left with a neutral zero line.
    """
    P = palette(dark)
    keys = [k for k in labels if k in scores]
    vals = [None if scores.get(k) is None or pd.isna(scores.get(k))
            else float(scores[k]) for k in keys]
    names = [labels[k] for k in keys]

    plot_vals = [0 if v is None else max(-3, min(3, v)) for v in vals]
    colours = [P["mid"] if v is None else (P["pos"] if v >= 0 else P["neg"])
               for v in vals]
    text = ["n/a" if v is None else f"{v:+.2f}" for v in vals]
    hover = []
    for k, v in zip(keys, vals):
        w = f"<br>Weight: {weights[k]*100:.0f}%" if weights and k in weights else ""
        hover.append(("No data for this stock" if v is None
                      else f"{v:+.2f} standard deviations vs today's universe") + w)

    fig = go.Figure(go.Bar(
        x=plot_vals, y=names, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=text, textposition="outside",
        textfont=dict(size=11, color=P["text2"]),
        hovertext=hover, hovertemplate="<b>%{y}</b><br>%{hovertext}<extra></extra>",
        showlegend=False, width=0.62,
    ))
    fig.add_vline(x=0, line=dict(color=P["axis"], width=1))
    _base_layout(fig, P, height)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(range=[-3.6, 3.6], showgrid=True, gridcolor=P["grid"],
                     zeroline=False, tickvals=[-3, -2, -1, 0, 1, 2, 3])
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(size=12, color=P["text2"]))
    return fig


# ---------------------------------------------------------------- distribution

def score_distribution(ranked: pd.DataFrame, highlight: Optional[List[str]] = None,
                       *, dark: bool = False, height: int = 220) -> go.Figure:
    """Where the picks sit in the full distribution of composite scores."""
    P = palette(dark)
    s = ranked["composite"].dropna()
    fig = go.Figure(go.Histogram(
        x=s, nbinsx=50, marker=dict(color=P["s1"], line=dict(width=0)),
        opacity=0.75, name="All scored stocks",
        hovertemplate="Score %{x:.2f}<br>%{y} stocks<extra></extra>",
    ))
    if highlight:
        for sym in highlight[:10]:
            if sym in ranked.index:
                v = ranked.loc[sym, "composite"]
                if pd.notna(v):
                    fig.add_vline(x=float(v), line=dict(color=P["s2"], width=1))
        fig.add_annotation(x=float(ranked.loc[highlight[0], "composite"]),
                           y=1, yref="paper", text="your picks", showarrow=False,
                           font=dict(size=11, color=P["s2"]),
                           xanchor="left", yanchor="top")
    _base_layout(fig, P, height, title="Composite score distribution")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text=None)
    return fig


# ------------------------------------------------------------------- backtest

def equity_curve(periods: pd.DataFrame, *, dark: bool = False,
                 height: int = 320) -> go.Figure:
    """Strategy vs benchmark, both indexed to 100 — one axis, comparable units."""
    P = palette(dark)
    eq = (1 + periods["portfolio"]).cumprod() * 100
    beq = (1 + periods["benchmark"].fillna(0)).cumprod() * 100
    x = pd.to_datetime(periods["date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=eq, name="Strategy", mode="lines",
                             line=dict(color=P["s1"], width=2),
                             hovertemplate="Strategy: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=beq, name="Benchmark", mode="lines",
                             line=dict(color=P["s2"], width=2),
                             hovertemplate="Benchmark: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=100, line=dict(color=P["axis"], width=1, dash="dot"))
    _base_layout(fig, P, height,
                 title="Growth of 100, after costs", legend=True, margin_top=62)
    fig.update_layout(legend=dict(y=1.03))
    return fig


def period_returns(periods: pd.DataFrame, *, dark: bool = False,
                   height: int = 220) -> go.Figure:
    """Per-period excess return over the benchmark — diverging by sign."""
    P = palette(dark)
    x = pd.to_datetime(periods["date"])
    ex = periods["excess"].fillna(0) * 100
    fig = go.Figure(go.Bar(
        x=x, y=ex,
        marker=dict(color=np.where(ex >= 0, P["pos"], P["neg"]), line=dict(width=0)),
        name="Excess return", showlegend=False,
        hovertemplate="%{x|%d %b %Y}<br>Excess: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=P["axis"], width=1))
    _base_layout(fig, P, height, title="Excess return vs benchmark, per period")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(ticksuffix="%")
    return fig


# --------------------------------------------------------------- market flows

def flow_chart(mf: pd.DataFrame, *, dark: bool = False,
               height: int = 260) -> go.Figure:
    """FII vs DII daily net cash flow — same units, so one axis is correct."""
    P = palette(dark)
    piv = mf[mf["metric"].isin(["FII_net_cr", "DII_net_cr"])].pivot_table(
        index="date", columns="metric", values="value")
    if piv.empty:
        return go.Figure()
    fig = go.Figure()
    for col, label, slot in (("FII_net_cr", "FII net", "s1"),
                             ("DII_net_cr", "DII net", "s2")):
        if col in piv.columns:
            fig.add_trace(go.Bar(
                x=piv.index, y=piv[col], name=label,
                marker=dict(color=P[slot], line=dict(width=0)), opacity=0.85,
                hovertemplate=label + ": ₹%{y:,.0f} cr<extra></extra>",
            ))
    fig.add_hline(y=0, line=dict(color=P["axis"], width=1))
    _base_layout(fig, P, height, title="Daily institutional cash flow (₹ crore)",
                 legend=True, margin_top=46)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return fig


def sector_strength(ranked: pd.DataFrame, sectors: Dict[str, str],
                    *, dark: bool = False, height: int = 340,
                    min_count: int = 3) -> go.Figure:
    """Median composite score by sector — diverging, since it's above/below zero."""
    P = palette(dark)
    df = ranked[["composite"]].copy()
    df["sector"] = [sectors.get(s, "Unclassified") for s in df.index]
    g = df.groupby("sector")["composite"].agg(["median", "count"])
    g = g[g["count"] >= min_count].sort_values("median")
    if g.empty:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=g["median"], y=g.index, orientation="h",
        marker=dict(color=np.where(g["median"] >= 0, P["pos"], P["neg"]),
                    line=dict(width=0)),
        customdata=g["count"],
        hovertemplate="<b>%{y}</b><br>Median score %{x:+.2f}"
                      "<br>%{customdata} stocks<extra></extra>",
        showlegend=False, width=0.62,
    ))
    fig.add_vline(x=0, line=dict(color=P["axis"], width=1))
    _base_layout(fig, P, height, title="Median composite score by sector")
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig
