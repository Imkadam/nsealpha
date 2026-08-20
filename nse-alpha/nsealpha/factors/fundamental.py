"""
fundamental.py — is the business behind the ticker any good, and is it cheap?

Four sub-blocks, each a composite of the ratios Indian investors actually quote:

  valuation   P/E relative to the stock's own sector P/E (an absolute P/E of 40
              means nothing without knowing whether the sector trades at 15 or
              60), P/B, PEG, EV/EBITDA, dividend yield. Cheap scores high — but
              *too* cheap is a trap, so a P/E under ~4 stops earning extra marks.
  quality     ROE and ROCE (the two numbers that separate a compounder from a
              capital incinerator), operating and net margin, debt-to-equity,
              current ratio. Indian mid-caps die of leverage, so D/E is weighted
              hard and anything above ~2x is heavily penalised.
  growth      YoY revenue growth, YoY/QoQ profit growth, EPS growth. Growth is
              only credited when it is not being bought with debt, so the growth
              score is scaled down when D/E is extreme.
  cash_quality Cash flow, from the actual cash-flow statement. Free cash flow
              (CFO - capex) and how many of the last five years it was positive;
              and the cash-conversion ratio, cumulative CFO over cumulative PAT.
              Reported profit is an opinion, cash is a fact — the classic Indian
              small-cap failure is profit that grows beautifully while operating
              cash flow does not follow, because receivables and inventory are
              absorbing it. Five years of CFO >= PAT is very hard to fake.
  ownership   Promoter holding (high and stable is good), promoter pledge (any
              pledge is bad, heavy pledge is disqualifying in spirit), and
              institutional holding.

Interest coverage (EBIT / interest) and gross margin now sit inside `quality`,
and ROCE is the real EBIT / capital-employed figure rather than the ROA x 1.4
stand-in used before statements were available. Banks and NBFCs are excluded
from interest coverage, gross margin and capex-derived metrics, because interest
is their raw material rather than a financing cost.

Missing data is left as NaN rather than filled with zero — a stock with no
fundamentals shouldn't be scored as if it had *average* fundamentals. scoring.py
handles coverage requirements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df, name):
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
        else pd.Series(np.nan, index=df.index)


def _score_lower_better(x, good, bad, floor=None):
    """Map a 'lower is better' ratio to 0..1. Below `floor` stops helping."""
    x = pd.to_numeric(x, errors="coerce")
    v = np.clip((bad - x) / (bad - good + 1e-9), 0, 1)
    if floor is not None:
        v = np.where(x < floor, np.clip(v * 0.6, 0, 1), v)   # value-trap discount
    return pd.Series(v, index=x.index)


def _score_higher_better(x, bad, good, cap=None):
    x = pd.to_numeric(x, errors="coerce")
    v = np.clip((x - bad) / (good - bad + 1e-9), 0, 1)
    if cap is not None:
        v = np.where(x > cap, np.clip(v * 0.85, 0, 1), v)    # implausibly good = suspicious
    return pd.Series(v, index=x.index)


def compute(ctx) -> pd.DataFrame:
    idx = ctx.snapshot.index
    out = pd.DataFrame(index=idx)

    f = ctx.fundamentals
    if f is None or f.empty:
        for c in ("valuation", "quality", "growth", "cash_quality", "ownership"):
            out[c] = np.nan
        return out

    f = f.set_index("symbol").reindex(idx)

    # Statement-derived metrics, when we have them. Merged in so the code below
    # can treat them like any other column; statement values take precedence over
    # the summary-ratio equivalents because they are computed, not scraped.
    fin = getattr(ctx, "financials", None)
    if fin is not None and not fin.empty:
        fin = fin.reindex(idx)
    else:
        fin = pd.DataFrame(index=idx)

    def fin_col(name):
        return (pd.to_numeric(fin[name], errors="coerce")
                if name in fin.columns else pd.Series(np.nan, index=idx))

    pe = _col(f, "pe")
    sector_pe = _col(f, "sector_pe")
    pb = _col(f, "pb")
    peg = _col(f, "peg")
    evebitda = _col(f, "ev_to_ebitda")
    divy = _col(f, "dividend_yield")

    roe = _col(f, "roe")
    roe = roe.fillna(fin_col("roe_stmt"))

    # Real ROCE = EBIT / (equity + long-term debt), computed from the statements.
    # The ROA x 1.4 stand-in below is now only a last resort for companies whose
    # statements we could not fetch, and it is flagged as such in the diagnostics.
    roce = fin_col("roce")
    roce_is_estimated = roce.isna()
    roce = roce.fillna(_col(f, "roce"))
    roce_is_estimated &= roce.isna()
    roce = roce.fillna(_col(f, "roa") * 1.4)

    opm = _col(f, "operating_margin")
    npm = _col(f, "profit_margin")
    gm = fin_col("gross_margin")
    icr = fin_col("interest_coverage")
    de = _col(f, "debt_to_equity")
    cr = _col(f, "current_ratio")

    rev_g = _col(f, "revenue_growth")
    eps_g = _col(f, "earnings_growth")
    pat_g = _col(f, "pat_growth_yoy")
    qoq_g = _col(f, "pat_growth_qoq")

    promoter = _col(f, "promoter_holding")
    if promoter.isna().all():
        promoter = _col(f, "held_by_insiders")
    pledge = _col(f, "promoter_pledge")
    instl = _col(f, "held_by_institutions")
    fii_chg = _col(f, "fii_change")

    # ------------------------------------------------------------- valuation
    # Relative P/E is the headline: below the sector's multiple is a discount.
    rel_pe = pe / sector_pe.replace(0, np.nan)
    val_rel = _score_lower_better(rel_pe, good=0.55, bad=1.8)
    val_pe = _score_lower_better(pe, good=12, bad=60, floor=4)
    val_pb = _score_lower_better(pb, good=1.2, bad=9, floor=0.4)
    val_peg = _score_lower_better(peg, good=0.6, bad=3.0, floor=0.15)
    val_ev = _score_lower_better(evebitda, good=7, bad=32, floor=2)
    val_dy = _score_higher_better(divy, bad=0.0, good=3.0)

    parts = pd.concat([val_rel * 2.0, val_pe * 1.2, val_pb * 1.0,
                       val_peg * 1.2, val_ev * 1.0, val_dy * 0.6], axis=1)
    wts = np.array([2.0, 1.2, 1.0, 1.2, 1.0, 0.6])
    out["valuation"] = _weighted_nanmean(parts, wts) * 5

    # ---------------------------------------------------------------- quality
    q_roe = _score_higher_better(roe, bad=6, good=25, cap=70)
    q_roce = _score_higher_better(roce, bad=8, good=25, cap=70)
    q_opm = _score_higher_better(opm, bad=4, good=22, cap=65)
    q_npm = _score_higher_better(npm, bad=2, good=15, cap=50)
    q_gm = _score_higher_better(gm, bad=12, good=45, cap=90)
    # Interest coverage: the guide's threshold is 4.0x. Below ~2x a cyclical
    # downturn stops the company servicing its debt, which is how Indian mid-caps
    # actually die; above ~12x the extra cover stops meaning much.
    q_icr = _score_higher_better(icr, bad=1.5, good=8.0, cap=200)
    # yfinance reports D/E in percent (e.g. 45.0 = 0.45x); normalise if so
    de_x = np.where(de > 10, de / 100.0, de)
    de_x = pd.Series(de_x, index=idx)
    q_de = _score_lower_better(de_x, good=0.15, bad=2.0)
    q_cr = _score_higher_better(cr, bad=0.8, good=2.0, cap=6)
    # Multi-year consistency: the guide asks for ROCE/ROE above 15% across 3-5
    # years, not one good year. Only scored where we have the history.
    q_consist = _score_higher_better(
        (fin_col("roce_consistency").fillna(0) * 0.5
         + fin_col("roe_consistency").fillna(0) * 0.5)
        .where(fin_col("roce_consistency").notna()
               | fin_col("roe_consistency").notna()),
        bad=0.2, good=0.9)

    parts = pd.concat([q_roe * 2.0, q_roce * 1.8, q_icr * 1.6, q_de * 2.0,
                       q_opm * 1.1, q_gm * 1.0, q_npm * 0.9, q_cr * 0.7,
                       q_consist * 1.4], axis=1)
    wts = np.array([2.0, 1.8, 1.6, 2.0, 1.1, 1.0, 0.9, 0.7, 1.4])
    out["quality"] = _weighted_nanmean(parts, wts) * 5

    # ----------------------------------------------------------------- growth
    g_rev = _score_higher_better(rev_g, bad=0, good=25, cap=120)
    g_eps = _score_higher_better(eps_g, bad=0, good=30, cap=200)
    g_pat = _score_higher_better(pat_g, bad=0, good=30, cap=200)
    g_qoq = _score_higher_better(qoq_g, bad=-5, good=20, cap=150)
    parts = pd.concat([g_rev * 1.6, g_eps * 1.5, g_pat * 1.4, g_qoq * 1.0], axis=1)
    growth = _weighted_nanmean(parts, np.array([1.6, 1.5, 1.4, 1.0]))
    # growth bought with leverage is worth less
    leverage_haircut = np.where(de_x > 2.0, 0.7, np.where(de_x > 1.0, 0.88, 1.0))
    out["growth"] = growth * pd.Series(leverage_haircut, index=idx) * 5

    # ------------------------------------------------------------ cash quality
    # The guide's two cash tests, and the reason they exist: a company can report
    # rising profit for years while operating cash flow goes nowhere, because the
    # "profit" is sitting in receivables and inventory. Cash conversion catches
    # that; nothing on the income statement does.
    conv = fin_col("cash_conversion")
    fcf_cum = fin_col("fcf_cumulative")
    fcf_share = fin_col("fcf_positive_share")
    fcf_margin = fin_col("fcf_margin")

    # CFO/PAT: 1.0 means profit fully converts to cash. Below ~0.6 is a warning;
    # above ~1.3 usually just means heavy depreciation, so it stops earning more.
    c_conv = _score_higher_better(conv, bad=0.4, good=1.2, cap=3.0)
    c_pos = _score_higher_better(fcf_share, bad=0.2, good=1.0)
    c_cum = pd.Series(np.where(np.isfinite(fcf_cum), (fcf_cum > 0).astype(float), np.nan),
                      index=idx)
    c_marg = _score_higher_better(fcf_margin, bad=-2.0, good=12.0, cap=45)

    parts = pd.concat([c_conv * 2.2, c_pos * 1.8, c_cum * 1.5, c_marg * 1.0], axis=1)
    cash_score = _weighted_nanmean(parts, np.array([2.2, 1.8, 1.5, 1.0]))
    # Discount thin histories rather than treating two years like five.
    yrs = fin_col("fcf_years").fillna(fin_col("cash_conversion_years"))
    confidence = np.clip(yrs.fillna(0) / 4.0, 0.4, 1.0)
    out["cash_quality"] = (cash_score * confidence) * 5

    # -------------------------------------------------------------- ownership
    o_prom = _score_higher_better(promoter, bad=25, good=62, cap=90)
    o_pledge = _score_lower_better(pledge.fillna(0), good=0.0, bad=25.0)
    o_inst = _score_higher_better(instl, bad=2, good=28, cap=75)
    o_fii = _score_higher_better(fii_chg, bad=-1.0, good=1.0)
    parts = pd.concat([o_prom * 1.4, o_pledge * 2.0, o_inst * 1.0, o_fii * 1.0], axis=1)
    out["ownership"] = _weighted_nanmean(parts, np.array([1.4, 2.0, 1.0, 1.0])) * 5

    # diagnostics surfaced on the website
    for name, series in (("_pe", pe), ("_sector_pe", sector_pe), ("_pb", pb),
                         ("_roe", roe), ("_roce", roce), ("_de", de_x),
                         ("_rev_growth", rev_g), ("_eps_growth", eps_g),
                         ("_div_yield", divy), ("_promoter", promoter),
                         ("_pledge", pledge), ("_mcap_cr", _col(f, "market_cap_cr")),
                         # statement-derived, surfaced so the UI can show them
                         ("_gross_margin", gm), ("_icr", icr),
                         ("_cash_conversion", conv), ("_fcf_cumulative", fcf_cum),
                         ("_fcf_positive_years", fin_col("fcf_positive_years")),
                         ("_fcf_years", fin_col("fcf_years")),
                         ("_roce_consistency", fin_col("roce_consistency")),
                         ("_roe_consistency", fin_col("roe_consistency")),
                         ("_fin_years", fin_col("fin_years")),
                         ("_roce_estimated", roce_is_estimated.astype(float))):
        out[name] = series
    return out


def _weighted_nanmean(parts: pd.DataFrame, wts: np.ndarray) -> pd.Series:
    """Weighted mean that ignores NaN components instead of poisoning the row."""
    vals = parts.to_numpy(dtype=float)
    # parts were already pre-multiplied by weights; recover the raw values
    raw = vals / wts[None, :]
    mask = ~np.isnan(raw)
    wsum = (mask * wts[None, :]).sum(axis=1)
    num = np.nansum(raw * wts[None, :], axis=1)
    res = np.where(wsum > 0, num / np.where(wsum == 0, 1, wsum), np.nan)
    return pd.Series(res, index=parts.index)
