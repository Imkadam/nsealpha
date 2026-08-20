"""
financials.py — annual statements, and the metrics the master guide asks for that
price-and-ratio data alone cannot give you.

WHY THIS MODULE EXISTS
----------------------
The guide's mandatory buy checklist names three things the project could not
compute before:

    Interest Coverage Ratio  > 4.0x       (EBIT / interest expense)
    Free Cash Flow           positive over 5 years   (CFO - capex)
    Cash-flow conversion     5-yr CFO >= PAT

All three need the actual cash-flow statement and balance sheet, not the summary
ratios in a quote endpoint. It also lets us fix two weaker things:

    ROCE   was approximated as ROA x 1.4 because neither NSE's public JSON nor
           yfinance's `info` dict publishes it. With the statements we can compute
           the real thing: EBIT / (equity + long-term debt).
    Gross margin  was missing entirely — we had operating and net margin, but not
           the COGS-level number the guide calls "product-level pricing power".

WHY CASH FLOW MATTERS MORE THAN THE RATIOS
------------------------------------------
Reported profit is an opinion; cash is a fact. The single most useful check in the
whole guide is CFO >= PAT over five years, because the classic Indian small-cap
accounting failure is revenue and profit that grow beautifully while operating
cash flow does not follow — receivables balloon, inventory piles up, and the
"profit" turns out to be an entry in a ledger. A company that has converted
profit into cash for five straight years is very hard to fake.

DATA SOURCE AND ITS LIMITS
--------------------------
yfinance's `income_stmt`, `balance_sheet` and `cashflow` (one request each per
symbol, so this runs on the filtered universe and refreshes monthly, not daily).

Honest caveats:
  * Yahoo typically returns FOUR annual periods, occasionally five. So "5-year
    FCF" is really "every year we have, up to five". Every metric below reports
    `years` alongside its value so you can see what it was computed from, and the
    scoring discounts thin histories rather than pretending.
  * These are consolidated figures as Yahoo has them, which for Indian companies
    can lag the exchange filing by weeks and occasionally mis-map a line item.
    Treat a single surprising number as a data question first.
  * Banks and NBFCs make ICR, capex and gross margin meaningless — interest is
    their raw material, not a financing cost. Those sectors are excluded from the
    affected metrics rather than scored on nonsense.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("nsealpha.financials")

# Sectors where interest expense is an operating input, not a financing cost, and
# where "capex" and "gross margin" have no sensible meaning.
FINANCIAL_SECTOR_PATTERNS = re.compile(
    r"financ|bank|nbfc|insur|capital market|asset manage", re.I)

MAX_YEARS = 5


# --------------------------------------------------------------- field lookup

def _norm(label: str) -> str:
    """'Operating Cash Flow' and 'OperatingCashFlow' must resolve to one key."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


# Each metric lists aliases in priority order. yfinance has shipped both the
# camel-case and the title-cased forms across versions, and renames line items
# occasionally, so we match on the normalised form and try several names.
FIELDS: Dict[str, List[str]] = {
    "revenue":        ["TotalRevenue", "OperatingRevenue"],
    "cogs":           ["CostOfRevenue", "ReconciledCostOfRevenue"],
    "gross_profit":   ["GrossProfit"],
    "ebitda":         ["EBITDA", "NormalizedEBITDA"],
    "ebit":           ["EBIT", "OperatingIncome", "TotalOperatingIncomeAsReported"],
    "interest":       ["InterestExpense", "InterestExpenseNonOperating"],
    "pretax":         ["PretaxIncome"],
    "pat":            ["NetIncome", "NetIncomeCommonStockholders",
                       "NetIncomeContinuousOperations"],
    "equity":         ["StockholdersEquity", "TotalEquityGrossMinorityInterest"],
    "total_debt":     ["TotalDebt"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"],
    "invested_capital": ["InvestedCapital"],
    "total_assets":   ["TotalAssets"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities"],
    "cfo":            ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"],
    "capex":          ["CapitalExpenditure", "CapitalExpenditureReported"],
    "fcf_reported":   ["FreeCashFlow"],
}


def _extract(statements: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Fold the three statement frames into one tidy frame:
    index = fiscal period (newest first), columns = the FIELDS keys above.
    """
    lookup: Dict[str, Dict[pd.Timestamp, float]] = {}
    for df in statements:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        norm_index = {_norm(ix): ix for ix in df.index}
        for key, aliases in FIELDS.items():
            if key in lookup:
                continue
            for alias in aliases:
                ix = norm_index.get(_norm(alias))
                if ix is None:
                    continue
                row = df.loc[ix]
                if isinstance(row, pd.DataFrame):     # duplicate labels
                    row = row.iloc[0]
                vals = {pd.Timestamp(c): pd.to_numeric(v, errors="coerce")
                        for c, v in row.items()}
                if any(pd.notna(v) for v in vals.values()):
                    lookup[key] = vals
                    break

    if not lookup:
        return pd.DataFrame()

    out = pd.DataFrame(lookup)
    out = out.sort_index(ascending=False)             # newest period first
    return out.head(MAX_YEARS + 1)


def fetch_statements_yf(symbol: str) -> pd.DataFrame:
    """One symbol's annual statements, tidied. Empty frame if unavailable."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()
    try:
        t = yf.Ticker(f"{symbol}.NS")
        return _extract([t.income_stmt, t.balance_sheet, t.cashflow])
    except Exception as e:
        log.debug("statements unavailable for %s: %s", symbol, e)
        return pd.DataFrame()


# ------------------------------------------------------------------- ingestion

def refresh_statements(store, symbols: Iterable[str], max_age_days: int = 30,
                       progress=None) -> int:
    """
    Annual statements change four times a year at most, so this refreshes monthly.
    One symbol per request, on the already-filtered universe only.
    """
    symbols = list(symbols)
    today = date.today().isoformat()
    done = 0
    for i, sym in enumerate(symbols):
        age = store.statements_age_days(sym)
        if age is not None and age < max_age_days:
            continue
        df = fetch_statements_yf(sym)
        if not df.empty:
            store.upsert_statements(sym, df, fetched=today)
            done += 1
        if progress and i % 25 == 0:
            progress(i + 1, len(symbols), f"statements {sym}")
    log.info("financial statements refreshed for %d symbols", done)
    return done


# --------------------------------------------------------------- the metrics

def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def compute_metrics(df: pd.DataFrame, is_financial: bool = False) -> Dict[str, float]:
    """
    Derive the guide's metrics from one company's tidy statement history.
    `df` is newest-period-first. Returns NaN rather than guessing.
    """
    if df is None or df.empty:
        return {}

    m: Dict[str, float] = {}
    latest = df.iloc[0]

    def val(col):
        v = latest.get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    revenue = val("revenue")
    ebit = val("ebit")
    ebitda = val("ebitda")
    pat = val("pat")
    interest = val("interest")
    equity = val("equity")
    ltd = val("long_term_debt")
    invested = val("invested_capital")
    gross = val("gross_profit")
    cogs = val("cogs")

    m["fin_years"] = int(len(df))

    # ---- margins -----------------------------------------------------------
    if not is_financial and np.isfinite(revenue) and revenue > 0:
        if not np.isfinite(gross) and np.isfinite(cogs):
            gross = revenue - cogs
        if np.isfinite(gross):
            m["gross_margin"] = 100.0 * gross / revenue
        if np.isfinite(ebitda):
            m["ebitda_margin"] = 100.0 * ebitda / revenue

    # ---- interest coverage -------------------------------------------------
    # Yahoo reports interest expense as a positive magnitude on the income
    # statement, but not always; take the absolute value either way. A company
    # with no debt has no interest line at all, which is the best possible
    # outcome and must not be read as "missing data".
    if not is_financial and np.isfinite(ebit):
        if np.isfinite(interest) and abs(interest) > 0:
            m["interest_coverage"] = ebit / abs(interest)
        elif np.isfinite(val("total_debt")) and val("total_debt") == 0:
            m["interest_coverage"] = 999.0      # debt-free
        elif "interest" not in df.columns and np.isfinite(val("total_debt")) \
                and val("total_debt") == 0:
            m["interest_coverage"] = 999.0

    # ---- true ROCE ---------------------------------------------------------
    # EBIT / capital employed, where capital employed = equity + long-term debt.
    # This replaces the old ROA x 1.4 approximation.
    if not is_financial and np.isfinite(ebit):
        capital = np.nan
        if np.isfinite(invested) and invested > 0:
            capital = invested
        elif np.isfinite(equity):
            capital = equity + (ltd if np.isfinite(ltd) else 0.0)
        if np.isfinite(capital) and capital > 0:
            m["roce"] = 100.0 * ebit / capital

    if np.isfinite(pat) and np.isfinite(equity) and equity > 0:
        m["roe_stmt"] = 100.0 * pat / equity

    # ---- free cash flow ----------------------------------------------------
    cfo_s = _series(df, "cfo")
    capex_s = _series(df, "capex")
    if len(cfo_s):
        # Yahoo signs capex negative (a cash outflow). Subtract the magnitude so
        # the arithmetic is right whichever sign convention arrives.
        capex_aligned = capex_s.reindex(cfo_s.index)
        fcf = cfo_s - capex_aligned.abs().fillna(0.0)
        fcf = fcf.head(MAX_YEARS)
        m["fcf_latest"] = float(fcf.iloc[0])
        m["fcf_cumulative"] = float(fcf.sum())
        m["fcf_years"] = int(len(fcf))
        m["fcf_positive_years"] = int((fcf > 0).sum())
        m["fcf_positive_share"] = float((fcf > 0).mean())
        if np.isfinite(revenue) and revenue > 0:
            m["fcf_margin"] = 100.0 * float(fcf.iloc[0]) / revenue

    # ---- cash conversion: cumulative CFO vs cumulative PAT -----------------
    pat_s = _series(df, "pat")
    if len(cfo_s) and len(pat_s):
        n = min(len(cfo_s), len(pat_s), MAX_YEARS)
        cfo_sum = float(cfo_s.head(n).sum())
        pat_sum = float(pat_s.head(n).sum())
        m["cfo_sum"] = cfo_sum
        m["pat_sum"] = pat_sum
        m["cash_conversion_years"] = int(n)
        if pat_sum > 0:
            m["cash_conversion"] = cfo_sum / pat_sum
        elif cfo_sum > 0:
            # profits negative but cash positive — unusual, and not a failure
            m["cash_conversion"] = 1.5

    # ---- multi-year consistency -------------------------------------------
    # The guide asks for ROCE > 15% and ROE > 15% "across the last 3 to 5 years",
    # not just today. With history we can finally answer that.
    eq_s = _series(df, "equity")
    ebit_s = _series(df, "ebit")
    ltd_s = _series(df, "long_term_debt")
    # ROCE consistency is gated on sector for the same reason the current ROCE
    # is: EBIT over capital employed is not a meaningful figure for a bank.
    # ROE consistency below is NOT gated — return on equity means the same thing
    # for a lender as for a manufacturer.
    if not is_financial and len(ebit_s) and len(eq_s):
        n = min(len(ebit_s), len(eq_s), MAX_YEARS)
        cap = eq_s.head(n).values + np.nan_to_num(
            ltd_s.reindex(eq_s.index).head(n).values, nan=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            roce_hist = 100.0 * ebit_s.head(n).values / np.where(cap > 0, cap, np.nan)
        ok = np.isfinite(roce_hist)
        if ok.sum() >= 2:
            m["roce_years"] = int(ok.sum())
            m["roce_years_above_15"] = int((roce_hist[ok] > 15).sum())
            m["roce_consistency"] = float((roce_hist[ok] > 15).mean())
            m["roce_min"] = float(np.nanmin(roce_hist[ok]))
    if len(pat_s) and len(eq_s):
        n = min(len(pat_s), len(eq_s), MAX_YEARS)
        with np.errstate(divide="ignore", invalid="ignore"):
            roe_hist = 100.0 * pat_s.head(n).values / np.where(
                eq_s.head(n).values > 0, eq_s.head(n).values, np.nan)
        ok = np.isfinite(roe_hist)
        if ok.sum() >= 2:
            m["roe_years"] = int(ok.sum())
            m["roe_years_above_15"] = int((roe_hist[ok] > 15).sum())
            m["roe_consistency"] = float((roe_hist[ok] > 15).mean())

    m["is_financial_sector"] = bool(is_financial)
    return m


def metrics_table(store, sector_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Per-symbol derived metrics for every company we hold statements for."""
    raw = store.load_statements()
    if raw.empty:
        return pd.DataFrame()

    sector_map = sector_map or {}
    rows = {}
    for sym, g in raw.groupby("symbol", sort=False):
        wide = g.pivot_table(index="period", columns="item", values="value",
                             aggfunc="first")
        wide.index = pd.to_datetime(wide.index, errors="coerce")
        wide = wide[wide.index.notna()].sort_index(ascending=False)
        if wide.empty:
            continue
        sector = str(sector_map.get(sym, "") or "")
        is_fin = bool(FINANCIAL_SECTOR_PATTERNS.search(sector))
        m = compute_metrics(wide, is_financial=is_fin)
        if m:
            rows[sym] = m
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "symbol"
    return out


# ------------------------------------------------------------- checklist view

def checklist_flags(m: pd.Series) -> Dict[str, Optional[bool]]:
    """
    The guide's numeric checklist criteria, evaluated per stock.
    None means "we don't have the data", which is deliberately distinct from False.
    """
    def tri(cond, available):
        return bool(cond) if available else None

    icr = m.get("interest_coverage", np.nan)
    fcf_cum = m.get("fcf_cumulative", np.nan)
    conv = m.get("cash_conversion", np.nan)
    roce_c = m.get("roce_consistency", np.nan)
    roe_c = m.get("roe_consistency", np.nan)

    return {
        "icr_above_4": tri(icr > 4.0, np.isfinite(icr)),
        "fcf_positive_cumulative": tri(fcf_cum > 0, np.isfinite(fcf_cum)),
        "cfo_covers_pat": tri(conv >= 1.0, np.isfinite(conv)),
        "roce_consistently_above_15": tri(roce_c >= 0.6, np.isfinite(roce_c)),
        "roe_consistently_above_15": tri(roe_c >= 0.6, np.isfinite(roe_c)),
    }
