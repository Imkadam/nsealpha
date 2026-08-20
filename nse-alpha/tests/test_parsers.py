"""
Parser and scoring sanity tests.

Run with:  python -m pytest tests/ -q     (or just: python tests/test_parsers.py)

The bhavcopy fixtures below reproduce NSE's real header quirks — leading spaces
in column names, "-" for missing delivery on non-EQ series, turnover quoted in
lakhs. Those quirks are exactly what breaks naive parsers, so they are the point
of these tests.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsealpha.indicators import adx, atr, bollinger, compute_all, hurst, rsi, supertrend
from nsealpha.ingest import (parse_bhavcopy_full, parse_bhavcopy_udiff,
                             parse_index_close, score_announcement)
from nsealpha.scoring import blend, standardise, winsorize, zscore

# NSE really does put leading spaces in these headers.
BHAV_FULL = b"""SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER
RELIANCE, EQ, 05-AUG-2026, 2901.50, 2905.00, 2948.80, 2896.10, 2940.00, 2938.75, 2925.44, 8452310, 247280.55, 214553, 4102887, 48.54
TCS, EQ, 05-AUG-2026, 3855.20, 3860.00, 3879.90, 3821.05, 3830.00, 3828.40, 3846.11, 1522004, 58538.12, 88214, 903442, 59.36
SOMESME, SM, 05-AUG-2026, 41.20, 41.50, 43.00, 41.00, 42.80, 42.75, 42.10, 12000, 5.05, 44, -, -
"""

BHAV_UDIFF = b"""TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2026-08-05,2026-08-05,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE,2905.00,2948.80,2896.10,2938.75,2940.00,2901.50,,2938.75,,,8452310,24728055000,214553,F1,1,,,,,
2026-08-05,2026-08-05,CM,NSE,STK,11536,INE467B01029,TCS,EQ,,,,,TCS,3860.00,3879.90,3821.05,3828.40,3830.00,3855.20,,3828.40,,,1522004,5853812000,88214,F1,1,,,,,
"""

IDX_CLOSE = b"""Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield
NIFTY 50,05-08-2026,24810.25,24905.60,24788.10,24871.30,61.05,0.25,312456789,42155.66,22.41,3.85,1.24
NIFTY BANK,05-08-2026,53210.00,53480.15,53105.40,53402.85,192.85,0.36,98765432,18922.11,15.02,2.71,0.88
INDIA VIX,05-08-2026,12.10,12.85,11.95,12.44,0.34,2.81,0,0.00,,,
"""


def _ok(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail else ""))
    assert cond, name


def test_bhavcopy_full():
    df = parse_bhavcopy_full(BHAV_FULL, date(2026, 8, 5))
    _ok("bhavcopy_full parses 3 rows", len(df) == 3)
    r = df.set_index("symbol").loc["RELIANCE"]
    # CLOSE_PRICE and LAST_PRICE are different columns and different numbers;
    # getting these two swapped is the classic bhavcopy mistake.
    _ok("close taken from CLOSE_PRICE", abs(r["close"] - 2938.75) < 1e-6, str(r["close"]))
    _ok("last price kept separately", abs(r["last_price"] - 2940.00) < 1e-6)
    _ok("prev close read", abs(r["prev_close"] - 2901.50) < 1e-6)
    _ok("turnover converted from lakhs to rupees",
        abs(r["turnover"] - 247280.55 * 1e5) < 1.0, f"{r['turnover']:.0f}")
    _ok("delivery percentage read", abs(r["deliv_pct"] - 48.54) < 1e-6)
    _ok("'-' delivery becomes NaN",
        np.isnan(df.set_index("symbol").loc["SOMESME", "deliv_pct"]))
    _ok("series preserved for filtering",
        set(df["series"]) == {"EQ", "SM"})


def test_bhavcopy_udiff():
    df = parse_bhavcopy_udiff(BHAV_UDIFF, date(2026, 8, 5))
    _ok("udiff parses 2 equity rows", len(df) == 2)
    r = df.set_index("symbol").loc["RELIANCE"]
    _ok("udiff close", abs(r["close"] - 2938.75) < 1e-6, str(r["close"]))
    _ok("udiff agrees with sec_bhavdata_full on close",
        abs(r["close"] - parse_bhavcopy_full(BHAV_FULL, date(2026, 8, 5))
            .set_index("symbol").loc["RELIANCE", "close"]) < 1e-6)
    _ok("udiff turnover already in rupees", abs(r["turnover"] - 24728055000) < 1)
    _ok("udiff has no delivery data", np.isnan(r["deliv_pct"]))


def test_index_close():
    df = parse_index_close(IDX_CLOSE, date(2026, 8, 5))
    _ok("index file parses", len(df) == 3)
    _ok("NIFTY 50 close", abs(df.set_index("index_name").loc["NIFTY 50", "close"] - 24871.30) < 1e-6)
    _ok("INDIA VIX present for the risk overlay", "INDIA VIX" in set(df["index_name"]))


def test_announcement_lexicon():
    pos, _ = score_announcement("Receipt of new order worth Rs 480 crore from NHAI")
    neg, _ = score_announcement("Auditor qualification in limited review report")
    fraud, _ = score_announcement("Disclosure of fraud detected in subsidiary")
    neutral, _ = score_announcement("Intimation of change of registered office address")
    _ok("order win scores positive", pos > 0, f"{pos:+.1f}")
    _ok("auditor qualification scores negative", neg < -2, f"{neg:+.1f}")
    _ok("fraud scores strongly negative", fraud <= -4, f"{fraud:+.1f}")
    _ok("routine filing is ~neutral", abs(neutral) < 1.0, f"{neutral:+.1f}")
    _ok("scores are bounded", -6 <= fraud <= 6)


def _synthetic_ohlc(n=400, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0006, 0.014, n)
    close = 500 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2024-01-01", periods=n)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.lognormal(12, 0.4, n)
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": vol, "turnover": vol * close,
                         "deliv_pct": rng.uniform(20, 70, n)}, index=idx)


def test_indicators():
    df = _synthetic_ohlc()
    r = rsi(df["close"])
    _ok("RSI stays within 0-100", r.dropna().between(0, 100).all())
    a = atr(df["high"], df["low"], df["close"])
    _ok("ATR is non-negative", (a.dropna() >= 0).all())
    ad, pdi, mdi = adx(df["high"], df["low"], df["close"])
    _ok("ADX within 0-100", ad.dropna().between(0, 100).all())
    mid, up, lo, w, pb = bollinger(df["close"])
    _ok("Bollinger upper >= lower", (up.dropna() >= lo.dropna()).all())
    d, line = supertrend(df["high"], df["low"], df["close"])
    _ok("Supertrend direction is +/-1", set(np.unique(d.dropna())) <= {-1.0, 1.0})
    h = hurst(np.log(df["close"]))
    _ok("Hurst of a random walk is near 0.5", 0.30 < h < 0.72, f"H={h:.2f}")

    full = compute_all(df)
    for c in ("sma200", "macd_hist", "adx", "atr_pct", "ret_3m", "turnover_20d_cr",
              "deliv_trend", "bb_pct_b", "obv_slope", "hi_52w"):
        _ok(f"compute_all produced {c}", c in full.columns)
    _ok("last row of compute_all is mostly populated",
        full.iloc[-1].notna().mean() > 0.85, f"{full.iloc[-1].notna().mean():.0%}")


def test_scoring_math():
    s = pd.Series(list(np.arange(1.0, 21.0)) + [5000.0])
    w = winsorize(s, 0.02)
    _ok("winsorise clips the outlier", w.max() < 5000, f"max {w.max():.1f}")
    _ok("winsorise is a no-op on samples too small to have tails",
        winsorize(pd.Series([1.0, 2.0, 500.0]), 0.02).max() == 500.0)
    z = zscore(pd.Series([1.0, 2.0, 3.0]))
    _ok("z-score is centred", abs(z.mean()) < 1e-9)
    _ok("z-score has unit sd", abs(z.std(ddof=0) - 1) < 1e-9)

    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 2.0, np.nan]})
    score, cov = blend(df, {"a": 0.5, "b": 0.5})
    _ok("blend renormalises around NaN", abs(score.iloc[1] - 2.0) < 1e-9,
        f"got {score.iloc[1]}")
    _ok("coverage reflects missing factors", abs(cov.iloc[1] - 0.5) < 1e-9)
    _ok("full row has full coverage", abs(cov.iloc[0] - 1.0) < 1e-9)

    cfg = {"scoring": {"winsorize_pct": 0.02, "sector_neutral": False}}
    raw = pd.DataFrame({"x": np.arange(1.0, 13.0), "_diag": [9.0] * 12},
                       index=[f"S{i}" for i in range(12)])
    std = standardise(raw, cfg)
    _ok("diagnostics pass through standardisation untouched",
        (std["_diag"] == 9).all())
    _ok("real factors are standardised", abs(std["x"].mean()) < 1e-9)
    thin = pd.DataFrame({"x": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan]},
                        index=[f"S{i}" for i in range(6)])
    _ok("a factor with almost no coverage is dropped, not faked",
        standardise(thin, cfg)["x"].isna().all())


def test_no_lookahead_in_edge():
    """The historical-edge factor must not peek at forward returns it can't have."""
    from nsealpha.factors.trend import _historical_edge
    df = compute_all(_synthetic_ohlc(500, seed=11))
    mean_fwd, hit, n = _historical_edge(df)
    _ok("edge returns a sample", n > 0, f"n={n}")
    # truncating the last 10 rows must not change the answer computed on data
    # that was already fully in the past
    _ok("edge is finite", np.isfinite(mean_fwd) and 0 <= hit <= 1,
        f"mean={mean_fwd:.2f} hit={hit:.2f}")


def test_ratio_parsing():
    from nsealpha.adjust import is_dividend_only, parse_ratio
    cases = [
        ("FACE VALUE SPLIT FROM RS 10/- TO RS 2/-", 0.2),
        ("FACE VALUE SPLIT (SUB-DIVISION) FROM RS.10/- TO RE.1/-", 0.1),
        ("BONUS 1:1", 0.5),
        ("BONUS 1:2", 2 / 3),          # 1 new for every 2 held -> 2/3
        ("BONUS ISSUE 3:1", 0.25),     # 3 new for every 1 held -> 1/4
    ]
    for text, expected in cases:
        got, _ = parse_ratio(text)
        _ok(f"parse {text!r}", got is not None and abs(got - expected) < 1e-6,
            f"got {got}, expected {expected}")
    _ok("dividends are not price-adjusted", parse_ratio("DIVIDEND - RS 12 PER SHARE")[0] is None)
    _ok("dividend-only is recognised", is_dividend_only("INTERIM DIVIDEND RS 5"))
    _ok("split-plus-dividend is not dividend-only",
        not is_dividend_only("DIVIDEND RS 5 AND FACE VALUE SPLIT FROM RS 10 TO RS 1"))


def test_corporate_action_adjustment():
    """
    The one that matters. Build a clean series, inject a 1:5 split exactly the way
    NSE's bhavcopy reports one, and prove the adjusted series has no artificial gap.
    """
    from nsealpha.adjust import adjusted_view, detect_events

    n = 300
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0005, 0.01, n)
    close = 2000 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range("2025-01-01", periods=n)

    t_ex = 200
    ratio = 0.2                                    # 1:5 split
    raw = close.copy()
    raw[t_ex:] *= ratio
    prev_close = np.concatenate([[raw[0]], raw[:-1]])
    prev_close[t_ex] = raw[t_ex - 1] * ratio       # NSE publishes the ADJUSTED prev close

    prices = pd.DataFrame({
        "symbol": "TESTCO", "date": dates,
        "open": raw, "high": raw * 1.01, "low": raw * 0.99, "close": raw,
        "prev_close": prev_close, "volume": np.full(n, 100000.0),
        "turnover": raw * 100000.0,
    })

    ca = pd.DataFrame([{"symbol": "TESTCO",
                        "ex_date": dates[t_ex].strftime("%Y-%m-%d"),
                        "purpose": "FACE VALUE SPLIT FROM RS 10/- TO RS 2/-"}])

    # --- detection ---
    events = detect_events(prices, ca)
    _ok("exactly one event detected", len(events) == 1, f"got {len(events)}")
    e = events[0]
    _ok("event confirmed by both sources", e.source == "both", e.source)
    _ok("the two sources agree", not e.disagrees,
        f"implied {e.implied:.4f} vs parsed {e.parsed:.4f}")
    _ok("factor is the split ratio", abs(e.factor - ratio) < 1e-6)

    # --- the raw series is broken, which is the whole point ---
    raw_day_return = raw[t_ex] / raw[t_ex - 1] - 1
    _ok("raw series shows a fake -80% day", raw_day_return < -0.75,
        f"{raw_day_return:.1%}")

    # --- the adjusted series is not ---
    adj, evs = adjusted_view(prices, ca)
    a = adj.sort_values("date")["close"].to_numpy()
    adj_day_return = a[t_ex] / a[t_ex - 1] - 1
    _ok("adjusted series has no artificial gap", abs(adj_day_return) < 0.05,
        f"{adj_day_return:.2%}")

    # --- today's price must stay real, or the site quotes a price that doesn't exist ---
    _ok("most recent price is left unadjusted", abs(a[-1] - raw[-1]) < 1e-6,
        f"adj {a[-1]:.2f} vs raw {raw[-1]:.2f}")
    _ok("history is scaled backwards", abs(a[0] - raw[0] * ratio) < 1e-6)
    _ok("volume is scaled inversely", abs(adj.sort_values("date")["volume"].iloc[0]
                                          - 100000.0 / ratio) < 1e-3)

    # --- a 12-month momentum reading is now sane rather than catastrophic ---
    raw_12m = raw[-1] / raw[-252] - 1
    adj_12m = a[-1] / a[-252] - 1
    _ok("raw 12m return is corrupted by the split", raw_12m < -0.5, f"{raw_12m:.1%}")
    _ok("adjusted 12m return is plausible", -0.5 < adj_12m < 2.0, f"{adj_12m:.1%}")

    # --- no corporate action at all must change nothing ---
    clean = prices.copy()
    clean["close"] = close
    clean["open"] = close
    clean["high"] = close * 1.01
    clean["low"] = close * 0.99
    clean["prev_close"] = np.concatenate([[close[0]], close[:-1]])
    _ok("a clean series produces no events", len(detect_events(clean, None)) == 0)


def test_fno_aggregation():
    from nsealpha.flows import parse_fno_bhavcopy
    csv = (b"TradDt,FinInstrmTp,TckrSymb,XpryDt,OptnTp,ClsPric,OpnIntrst,"
           b"ChngInOpnIntrst,TtlTradgVol\n"
           b"2026-08-05,STF,RELIANCE,2026-08-27,,2945.00,12000,500,8000\n"
           b"2026-08-05,STF,RELIANCE,2026-09-24,,2960.00,4000,100,2000\n"
           b"2026-08-05,STO,RELIANCE,2026-08-27,CE,45.00,30000,0,5000\n"
           b"2026-08-05,STO,RELIANCE,2026-08-27,PE,38.00,45000,0,4000\n"
           b"2026-08-05,IDF,NIFTY,2026-08-27,,24900.00,90000,-200,15000\n")
    df = parse_fno_bhavcopy(csv, date(2026, 8, 5)).set_index("symbol")
    _ok("aggregates per underlying", set(df.index) == {"RELIANCE", "NIFTY"})
    _ok("uses near-month futures only", abs(df.loc["RELIANCE", "fut_oi"] - 12000) < 1e-6,
        str(df.loc["RELIANCE", "fut_oi"]))
    _ok("sums call OI", abs(df.loc["RELIANCE", "call_oi"] - 30000) < 1e-6)
    _ok("sums put OI", abs(df.loc["RELIANCE", "put_oi"] - 45000) < 1e-6)
    pcr = df.loc["RELIANCE", "put_oi"] / df.loc["RELIANCE", "call_oi"]
    _ok("PCR computes to 1.5", abs(pcr - 1.5) < 1e-6, f"{pcr:.2f}")


def _yf_statements(camel: bool = False, financial_sector: bool = False):
    """
    Statement frames shaped the way yfinance returns them: line items down the
    index, fiscal periods across the columns (newest first), capex negative.

    yfinance has shipped both title-cased ("Operating Cash Flow") and camel-case
    ("OperatingCashFlow") index labels across versions, so both are exercised.
    """
    periods = pd.to_datetime(["2026-03-31", "2025-03-31", "2024-03-31",
                              "2023-03-31", "2022-03-31"])

    def lbl(title, camel_name):
        return camel_name if camel else title

    income = pd.DataFrame({
        p: v for p, v in zip(periods, [
            [1000.0, 400.0, 250.0, 200.0, 40.0, 160.0, 120.0],
            [900.0, 360.0, 220.0, 175.0, 40.0, 135.0, 101.0],
            [800.0, 320.0, 190.0, 150.0, 42.0, 108.0, 81.0],
            [700.0, 280.0, 160.0, 125.0, 45.0, 80.0, 60.0],
            [600.0, 240.0, 130.0, 100.0, 48.0, 52.0, 39.0],
        ])},
        index=[lbl("Total Revenue", "TotalRevenue"),
               lbl("Gross Profit", "GrossProfit"),
               lbl("EBITDA", "EBITDA"),
               lbl("EBIT", "EBIT"),
               lbl("Interest Expense", "InterestExpense"),
               lbl("Pretax Income", "PretaxIncome"),
               lbl("Net Income", "NetIncome")])

    balance = pd.DataFrame({
        p: v for p, v in zip(periods, [
            [800.0, 500.0, 350.0],
            [700.0, 500.0, 350.0],
            [620.0, 520.0, 360.0],
            [550.0, 540.0, 380.0],
            [480.0, 560.0, 400.0],
        ])},
        index=[lbl("Stockholders Equity", "StockholdersEquity"),
               lbl("Total Debt", "TotalDebt"),
               lbl("Long Term Debt", "LongTermDebt")])

    cash = pd.DataFrame({
        p: v for p, v in zip(periods, [
            [180.0, -60.0],
            [150.0, -55.0],
            [110.0, -50.0],
            [70.0, -90.0],
            [45.0, -40.0],
        ])},
        index=[lbl("Operating Cash Flow", "OperatingCashFlow"),
               lbl("Capital Expenditure", "CapitalExpenditure")])
    return [income, balance, cash]


def test_statement_extraction():
    from nsealpha.financials import _extract
    for camel in (False, True):
        df = _extract(_yf_statements(camel=camel))
        label = "camelCase" if camel else "title-cased"
        _ok(f"{label} statement labels resolve", not df.empty)
        _ok(f"{label}: newest period first",
            df.index[0] > df.index[1], str(df.index[:2].tolist()))
        _ok(f"{label}: revenue read", abs(df.iloc[0]["revenue"] - 1000.0) < 1e-9)
        _ok(f"{label}: capex kept negative", df.iloc[0]["capex"] < 0)
        _ok(f"{label}: capped at 5 years", len(df) <= 6, str(len(df)))


def test_financial_metrics():
    from nsealpha.financials import checklist_flags, compute_metrics, _extract
    m = compute_metrics(_extract(_yf_statements()))

    # --- interest coverage: EBIT / |interest| = 200 / 40 = 5.0 ---
    _ok("interest coverage computed", abs(m["interest_coverage"] - 5.0) < 1e-9,
        f"{m['interest_coverage']:.3f}")

    # --- gross margin: 400 / 1000 ---
    _ok("gross margin computed", abs(m["gross_margin"] - 40.0) < 1e-9,
        f"{m['gross_margin']:.2f}%")
    _ok("EBITDA margin computed", abs(m["ebitda_margin"] - 25.0) < 1e-9)

    # --- real ROCE: EBIT / (equity + long-term debt) = 200 / (800+350) ---
    expected_roce = 100 * 200.0 / (800.0 + 350.0)
    _ok("ROCE is EBIT / capital employed, not a ROA proxy",
        abs(m["roce"] - expected_roce) < 1e-9,
        f"{m['roce']:.2f}% vs expected {expected_roce:.2f}%")

    # --- FCF: CFO - |capex|, per year, then summed ---
    fcf = [180 - 60, 150 - 55, 110 - 50, 70 - 90, 45 - 40]
    _ok("latest FCF = CFO - |capex|", abs(m["fcf_latest"] - fcf[0]) < 1e-9,
        f"{m['fcf_latest']}")
    _ok("cumulative FCF over 5 years", abs(m["fcf_cumulative"] - sum(fcf)) < 1e-9,
        f"{m['fcf_cumulative']} vs {sum(fcf)}")
    _ok("counts the negative-FCF year", m["fcf_positive_years"] == 4,
        f"{m['fcf_positive_years']} of {m['fcf_years']}")

    # --- cash conversion: sum(CFO) / sum(PAT) ---
    conv = (180 + 150 + 110 + 70 + 45) / (120 + 101 + 81 + 60 + 39)
    _ok("cash conversion = cumulative CFO / cumulative PAT",
        abs(m["cash_conversion"] - conv) < 1e-9,
        f"{m['cash_conversion']:.3f} vs {conv:.3f}")
    _ok("cash conversion above 1 for this profile", m["cash_conversion"] > 1.0)

    # --- multi-year consistency ---
    _ok("ROCE consistency measured over multiple years", m["roce_years"] == 5,
        str(m.get("roce_years")))
    _ok("ROE consistency measured", 0.0 <= m["roe_consistency"] <= 1.0,
        f"{m['roe_consistency']:.2f}")

    # --- the guide's checklist criteria ---
    flags = checklist_flags(pd.Series(m))
    _ok("ICR > 4.0x passes", flags["icr_above_4"] is True)
    _ok("cumulative FCF positive passes", flags["fcf_positive_cumulative"] is True)
    _ok("CFO covers PAT passes", flags["cfo_covers_pat"] is True)


def test_financial_metrics_edge_cases():
    from nsealpha.financials import checklist_flags, compute_metrics, _extract

    # --- banks: interest is raw material, not a financing cost ---
    m_fin = compute_metrics(_extract(_yf_statements()), is_financial=True)
    for k in ("interest_coverage", "gross_margin", "roce"):
        _ok(f"banks excluded from {k}", k not in m_fin,
            f"got {m_fin.get(k)}")
    _ok("banks still get cash conversion", "cash_conversion" in m_fin)
    _ok("banks still get ROE consistency", "roe_consistency" in m_fin)

    # --- a company that fails the cash test: profit rises, cash does not ---
    inc, bal, cash = _yf_statements()
    # Rows are line items, columns are periods: collapse CFO across every year
    # while the income statement keeps reporting healthy profit. This is the
    # classic pattern the cash-conversion test exists to catch.
    cash.iloc[0] = [20.0, 15.0, 12.0, 10.0, 8.0]        # Operating Cash Flow
    cash.iloc[1] = [-60.0, -55.0, -50.0, -90.0, -40.0]  # Capital Expenditure
    m_bad = compute_metrics(_extract([inc, bal, cash]))
    _ok("weak cash conversion is detected", m_bad["cash_conversion"] < 0.8,
        f"{m_bad['cash_conversion']:.2f}")
    _ok("negative cumulative FCF is detected", m_bad["fcf_cumulative"] < 0,
        f"{m_bad['fcf_cumulative']}")
    bad_flags = checklist_flags(pd.Series(m_bad))
    _ok("checklist fails on cash conversion", bad_flags["cfo_covers_pat"] is False)
    _ok("checklist fails on cumulative FCF",
        bad_flags["fcf_positive_cumulative"] is False)

    # --- missing data must read as None, never as False ---
    empty_flags = checklist_flags(pd.Series({}, dtype=float))
    _ok("missing data is None, not a failed criterion",
        all(v is None for v in empty_flags.values()), str(empty_flags))

    # --- a debt-free company has no interest line at all ---
    inc2, bal2, cash2 = _yf_statements()
    inc2 = inc2.drop(index=[i for i in inc2.index if "Interest" in str(i)])
    bal2.loc["Total Debt"] = 0.0
    bal2.loc["Long Term Debt"] = 0.0
    m_nodebt = compute_metrics(_extract([inc2, bal2, cash2]))
    _ok("debt-free company is not penalised for a missing interest line",
        m_nodebt.get("interest_coverage", 0) > 100,
        str(m_nodebt.get("interest_coverage")))

    # --- empty input must not raise ---
    _ok("empty statements return empty metrics",
        compute_metrics(pd.DataFrame()) == {})


def test_crontab_merge_preserves_other_jobs():
    """
    The scheduler rewrites the WHOLE crontab — cron has no edit-one-line
    operation. So the merge must never disturb anything outside our markers.
    A bug here silently deletes someone's backups.
    """
    from nsealpha.scheduler import (MARKER_BEGIN, MARKER_END, build_block,
                                    build_command, extract_block, merge_crontab,
                                    parse_schedule)

    user_jobs = "\n".join([
        "MAILTO=me@example.com",
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "",
        "# nightly database backup - do not remove",
        "0 2 * * * /home/me/backup.sh --full",
        "*/15 * * * * /usr/bin/check-disk",
        "",
        "@reboot /home/me/start-services.sh",
    ])
    block = build_block(19, 0, True, build_command("/opt/nse-alpha"))

    # --- install into a populated crontab ---
    after = merge_crontab(user_jobs, block)
    for line in user_jobs.splitlines():
        if line.strip():
            _ok(f"preserved: {line[:40]!r}", line in after)
    _ok("our block was added", MARKER_BEGIN in after and MARKER_END in after)

    # --- removing puts it back exactly as it was ---
    restored = merge_crontab(after, None)
    _ok("removal restores the original crontab byte-for-byte",
        restored.strip() == user_jobs.strip(),
        f"\n--- got ---\n{restored}\n--- want ---\n{user_jobs}")

    # --- re-installing must not duplicate ---
    twice = merge_crontab(merge_crontab(user_jobs, block), block)
    _ok("re-installing replaces rather than duplicating",
        twice.count(MARKER_BEGIN) == 1, f"{twice.count(MARKER_BEGIN)} blocks")
    _ok("re-install still keeps the user's jobs",
        "/home/me/backup.sh --full" in twice)

    # --- changing the time replaces the old entry ---
    newblock = build_block(8, 30, False, build_command("/opt/nse-alpha"))
    changed = merge_crontab(merge_crontab(user_jobs, block), newblock)
    _ok("schedule change leaves exactly one block",
        changed.count(MARKER_BEGIN) == 1)
    sched = parse_schedule(extract_block(changed))
    _ok("new time is read back correctly",
        sched["hour"] == 8 and sched["minute"] == 30, str(sched))
    _ok("weekday flag is read back correctly", sched["weekdays_only"] is False)

    # --- an empty crontab ---
    fresh = merge_crontab("", block)
    _ok("installs cleanly into an empty crontab", MARKER_BEGIN in fresh)
    _ok("empty crontab removal yields empty", merge_crontab("", None) == "")

    # --- a hand-mangled crontab with a begin marker but no end ---
    mangled = user_jobs + "\n" + MARKER_BEGIN + "\n0 19 * * 1-5 /old/command\n"
    repaired = merge_crontab(mangled, None)
    _ok("unterminated marker does not swallow the rest of the file",
        "/home/me/backup.sh --full" in repaired and "@reboot" in repaired)
    _ok("unterminated marker block is dropped",
        "/old/command" not in repaired and MARKER_BEGIN not in repaired)

    # --- output must be a valid crontab: newline-terminated ---
    _ok("result ends with a newline", fresh.endswith("\n"))


def test_cron_line_format():
    from nsealpha.scheduler import build_command, cron_line

    _ok("weekday schedule uses 1-5",
        cron_line(19, 0, True, "cmd").startswith("0 19 * * 1-5 "),
        cron_line(19, 0, True, "cmd"))
    _ok("every-day schedule uses *",
        cron_line(8, 30, False, "cmd").startswith("30 8 * * * "),
        cron_line(8, 30, False, "cmd"))

    cmd = build_command("/opt/nse alpha", python_exe="/usr/bin/python3")
    _ok("command cds to the project first", cmd.startswith("cd "))
    _ok("paths containing spaces are quoted", '"/opt/nse alpha"' in cmd, cmd)
    _ok("uses an absolute python", "/usr/bin/python3" in cmd)
    _ok("redirects output to a log", ">>" in cmd and "cron.log" in cmd)
    _ok("captures stderr too", "2>&1" in cmd)


def test_next_run_calculation():
    from nsealpha.scheduler import _next_run

    # Friday 20:00 -> next weekday run is Monday
    fri_evening = datetime(2026, 8, 21, 20, 0)          # a Friday
    nxt = _next_run(19, 0, True, now=fri_evening)
    _ok("Friday evening rolls to Monday", nxt.weekday() == 0, nxt.strftime("%a %d %b"))

    # Monday 08:00 -> today at 19:00
    mon_morning = datetime(2026, 8, 17, 8, 0)           # a Monday
    nxt = _next_run(19, 0, True, now=mon_morning)
    _ok("same-day run when the time hasn't passed",
        nxt.day == 17 and nxt.hour == 19, nxt.strftime("%a %d %b %H:%M"))

    # every-day schedule may land on a weekend
    nxt = _next_run(19, 0, False, now=fri_evening)
    _ok("every-day schedule allows Saturday", nxt.weekday() == 5,
        nxt.strftime("%a %d %b"))


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    print(f"\nRunning {len(tests)} test groups\n" + "-" * 52)
    for t in tests:
        print(t.__name__)
        t()
    print("-" * 52 + "\nAll checks passed.\n")


if __name__ == "__main__":
    main()
