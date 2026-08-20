"""
simulate.py — a synthetic NSE, for testing the pipeline without touching the network.

Why this exists: the scoring engine has a lot of moving parts, and you want to
know it works *before* you point it at two years of real bhavcopies. This module
manufactures a market with realistic structure — sectors with common factors,
per-stock idiosyncratic drift, volatility clustering, delivery percentages that
correlate with genuine accumulation, corporate announcements, bulk deals and
fundamentals — and loads it into the same SQLite tables the real ingest writes to.

Run `python run_daily.py --simulate` and the whole pipeline executes end to end.
The numbers are fiction. The plumbing is identical to production.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

SECTORS = ["Financial Services", "Information Technology", "Oil Gas & Consumable Fuels",
           "Fast Moving Consumer Goods", "Automobile and Auto Components",
           "Healthcare", "Metals & Mining", "Capital Goods", "Power",
           "Construction Materials", "Realty", "Chemicals"]

_PREFIX = ["ADANI", "BAJAJ", "TATA", "HIND", "BHARAT", "MAHA", "SHREE", "JINDAL",
           "GODREJ", "APOLLO", "TORRENT", "AARTI", "BALAJI", "DEEPAK", "SUPREME",
           "KIRLOSKAR", "FINOLEX", "VARUN", "NAVIN", "RATNA", "SANGHVI", "VISHAL"]
_SUFFIX = ["TECH", "FIN", "IND", "STEEL", "POWER", "CHEM", "MOTORS", "PHARMA",
           "CEMENT", "INFRA", "AGRO", "LABS", "AUTO", "ENERGY", "BANK", "PACK"]


def _symbols(n: int, rng) -> list:
    out, seen = [], set()
    while len(out) < n:
        s = rng.choice(_PREFIX) + rng.choice(_SUFFIX)
        if rng.random() < 0.25:
            s += str(rng.integers(1, 9))
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def generate(store, n_symbols: int = 350, n_days: int = 620, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    syms = _symbols(n_symbols, rng)

    end = date.today()
    days = []
    d = end
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days = sorted(days)
    dates = pd.to_datetime(days)

    T = len(dates)

    # --- market + sector factor structure -----------------------------------
    mkt = rng.normal(0.0004, 0.0090, T)
    # a couple of regime shifts so the risk overlay has something to detect
    mkt[int(T * 0.45):int(T * 0.55)] -= 0.0035
    mkt[int(T * 0.75):] += 0.0008
    sector_rets = {s: rng.normal(0, 0.0075, T) + rng.normal(0.0002, 0.0004) for s in SECTORS}

    price_rows, fund_rows, ann_rows, deal_rows = [], [], [], []
    ca_rows, fno_rows, sh_rows, fno_syms = [], [], [], []
    stmt_rows = []

    for sym in syms:
        sec = str(rng.choice(SECTORS))
        beta = float(rng.normal(1.0, 0.35))
        alpha = float(rng.normal(0.00025, 0.00075))
        idio_vol = float(np.clip(rng.lognormal(np.log(0.016), 0.45), 0.006, 0.055))
        quality = float(rng.beta(2.2, 2.2))          # latent "goodness" 0..1

        # volatility clustering
        vol = np.zeros(T)
        vol[0] = idio_vol
        for t in range(1, T):
            vol[t] = np.sqrt(0.90 * vol[t - 1] ** 2 +
                             0.10 * (rng.normal(0, idio_vol)) ** 2)
        eps = rng.normal(0, 1, T) * vol

        drift = alpha + (quality - 0.5) * 0.0009
        rets = drift + beta * mkt + 0.6 * sector_rets[sec] + eps

        # occasional gap events (results, orders)
        n_gaps = rng.poisson(4)
        for _ in range(n_gaps):
            t = int(rng.integers(30, T))
            rets[t] += rng.normal(0.0, 0.055)

        p0 = float(np.exp(rng.normal(np.log(320), 1.1)))
        close = p0 * np.exp(np.cumsum(rets))
        close = np.clip(close, 3, None)

        intraday = np.abs(rng.normal(0, 0.011, T)) + 0.004
        high = close * (1 + intraday)
        low = close * (1 - intraday * rng.uniform(0.5, 1.4, T))
        openp = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.005, T))
        openp = np.clip(openp, low, high)
        prev_close = np.concatenate([[close[0]], close[:-1]])

        base_vol = float(np.exp(rng.normal(np.log(400000), 1.3)))
        volume = base_vol * np.exp(rng.normal(0, 0.5, T)) * (1 + 3 * np.abs(rets))
        turnover = volume * close

        # delivery %: higher for quality names, rises when the stock is being
        # accumulated (positive 20d drift), noisy
        drift20 = pd.Series(rets).rolling(20, min_periods=1).mean().to_numpy()
        deliv = (25 + 35 * quality + 900 * drift20 + rng.normal(0, 6, T))
        deliv = np.clip(deliv, 4, 96)

        # --- corporate actions ------------------------------------------------
        # Roughly one stock in eight does a split or bonus in the window. The raw
        # price series gets the real overnight drop, and PREV_CLOSE on the ex-date
        # gets the ADJUSTED previous close, exactly as NSE publishes it. That is
        # what adjust.py has to detect and undo.
        if rng.random() < 0.12 and T > 200:
            t_ex = int(rng.integers(80, T - 40))
            kind = rng.choice(["split", "bonus"])
            if kind == "split":
                ratio = float(rng.choice([0.2, 0.5, 0.1]))       # 1:5, 1:2, 1:10
                fv_from = {0.2: 10, 0.5: 10, 0.1: 10}[ratio]
                fv_to = int(round(fv_from * ratio))
                purpose = f"FACE VALUE SPLIT FROM RS {fv_from}/- TO RS {fv_to}/-"
            else:
                a, b = rng.choice([(1, 1), (1, 2), (2, 1), (3, 1)])
                ratio = b / (a + b)
                purpose = f"BONUS {a}:{b}"
            # apply the real price drop from the ex-date onwards
            close[t_ex:] *= ratio
            high[t_ex:] *= ratio
            low[t_ex:] *= ratio
            openp[t_ex:] *= ratio
            volume[t_ex:] /= ratio
            turnover = volume * close
            prev_close = np.concatenate([[close[0]], close[:-1]])
            # NSE publishes the ADJUSTED previous close on the ex-date
            prev_close[t_ex] = close[t_ex - 1] * ratio
            ca_rows.append({"symbol": sym,
                            "ex_date": dates[t_ex].strftime("%Y-%m-%d"),
                            "purpose": purpose, "record_date": ""})

        price_rows.append(pd.DataFrame({
            "symbol": sym, "date": dates, "series": "EQ",
            "open": openp, "high": high, "low": low, "close": close,
            "prev_close": prev_close, "last_price": close,
            "volume": volume.round(), "turnover": turnover,
            "trades": (volume / 180).round(),
            "deliv_qty": (volume * deliv / 100).round(), "deliv_pct": deliv.round(2),
        }))

        # --- derivatives (only the more liquid names, as on the real exchange) --
        if base_vol > 300000 and close[-1] > 60:
            fno_syms.append(sym)
            oi_base = base_vol * rng.uniform(2, 8)
            # OI drifts with price for good names, against it for bad ones
            oi_walk = oi_base * np.exp(np.cumsum(
                rng.normal(0, 0.02, T) + drift20 * rng.uniform(-4, 12)))
            pcr_walk = np.clip(rng.lognormal(np.log(0.9 + 0.5 * quality), 0.25, T), 0.2, 3.0)
            basis = close * (1 + rng.normal(0.0009 * (quality - 0.35), 0.0016, T))
            keep = slice(max(0, T - 90), T)      # only recent history is needed
            fno_rows.append(pd.DataFrame({
                "symbol": sym, "date": dates[keep],
                "fut_close": basis[keep],
                "fut_oi": oi_walk[keep].round(),
                "fut_oi_chg": np.diff(oi_walk, prepend=oi_walk[0])[keep].round(),
                "fut_volume": (volume[keep] * rng.uniform(0.3, 1.5)).round(),
                "call_oi": (oi_walk[keep] * rng.uniform(0.6, 1.4)).round(),
                "put_oi": (oi_walk[keep] * rng.uniform(0.6, 1.4) * pcr_walk[keep]).round(),
                "call_volume": (volume[keep] * 0.4).round(),
                "put_volume": (volume[keep] * 0.3).round(),
            }))
            # quarterly shareholding, four quarters back
            prom = float(np.clip(rng.normal(30 + 35 * quality, 14), 0, 80))
            fii0 = float(np.clip(rng.normal(6 + 20 * quality, 8), 0, 60))
            dii0 = float(np.clip(rng.normal(5 + 14 * quality, 6), 0, 50))
            pl0 = float(max(0, rng.normal(14 - 16 * quality, 9)))
            for qi, qlabel in enumerate(["31-Dec-2025", "31-Mar-2026", "30-Jun-2026"]):
                sh_rows.append({
                    "symbol": sym, "period": qlabel,
                    "promoter": prom + rng.normal((quality - 0.5) * 0.4, 0.3) * qi,
                    "pledge": max(0.0, pl0 - (quality - 0.5) * 1.2 * qi),
                    "fii": max(0.0, fii0 + (quality - 0.45) * 1.5 * qi),
                    "dii": max(0.0, dii0 + (quality - 0.5) * 1.0 * qi),
                    "public": 100 - prom - fii0 - dii0,
                })

        # --- fundamentals consistent with the latent quality ----------------
        sector_pe = float(np.clip(rng.normal(24, 7), 9, 55))
        pe = float(np.clip(sector_pe * rng.lognormal(-0.15 + (0.5 - quality) * 0.5, 0.4), 3, 130))
        fund_rows.append((sym, {
            "source": "simulated", "company": sym.title() + " Limited",
            "sector": sec, "industry": sec,
            "pe": pe, "sector_pe": sector_pe,
            "pb": float(np.clip(rng.lognormal(np.log(2.4), 0.7) * (0.6 + quality), 0.2, 22)),
            "peg": float(np.clip(rng.lognormal(np.log(1.3), 0.6) * (1.5 - quality), 0.1, 8)),
            "ev_to_ebitda": float(np.clip(rng.normal(15, 7) * (1.3 - 0.5 * quality), 2, 60)),
            "roe": float(np.clip(rng.normal(6 + 26 * quality, 6), -22, 62)),
            "roce": float(np.clip(rng.normal(8 + 24 * quality, 6), -18, 65)),
            "operating_margin": float(np.clip(rng.normal(4 + 24 * quality, 7), -25, 55)),
            "profit_margin": float(np.clip(rng.normal(1 + 16 * quality, 5), -30, 40)),
            "debt_to_equity": float(np.clip(rng.lognormal(np.log(0.7), 0.9) * (1.6 - quality), 0, 6)),
            "current_ratio": float(np.clip(rng.normal(1.5 + quality, 0.5), 0.3, 5)),
            "revenue_growth": float(np.clip(rng.normal(4 + 22 * quality, 14), -40, 95)),
            "earnings_growth": float(np.clip(rng.normal(2 + 30 * quality, 26), -80, 190)),
            "pat_growth_yoy": float(np.clip(rng.normal(3 + 28 * quality, 24), -80, 180)),
            "pat_growth_qoq": float(np.clip(rng.normal(1 + 14 * quality, 18), -70, 120)),
            "dividend_yield": float(np.clip(rng.gamma(1.6, 0.8) * quality, 0, 7)),
            "promoter_holding": float(np.clip(rng.normal(30 + 35 * quality, 14), 0, 80)),
            "promoter_pledge": float(max(0, rng.normal(14 - 16 * quality, 9))),
            "held_by_institutions": float(np.clip(rng.normal(6 + 26 * quality, 10), 0, 70)),
            "fii_change": float(rng.normal((quality - 0.5) * 1.2, 0.8)),
            "market_cap_cr": float(close[-1] * np.exp(rng.normal(np.log(4.0e7), 1.4)) / 1e7),
        }))

        # --- annual financial statements --------------------------------------
        # Built so the derived metrics are internally consistent: a high-quality
        # company gets good interest cover, positive free cash flow and CFO that
        # exceeds PAT, while a low-quality one shows the classic pattern of
        # profit that never turns into cash.
        rev0 = float(np.exp(rng.normal(np.log(2.2e9), 1.3)))
        gm_pct = np.clip(rng.normal(18 + 30 * quality, 9), 5, 78) / 100
        ebitda_pct = np.clip(rng.normal(6 + 22 * quality, 6), 1, 45) / 100
        debt = rev0 * np.clip(rng.lognormal(np.log(0.35), 0.8) * (1.5 - quality), 0, 3)
        equity0 = rev0 * np.clip(rng.normal(0.55 + 0.5 * quality, 0.25), 0.08, 3)
        # cash conversion: good companies convert >1, weak ones leak into working capital
        conv = float(np.clip(rng.normal(0.55 + 0.75 * quality, 0.22), 0.05, 1.9))
        for yr_back in range(5):
            period = date(end.year - yr_back, 3, 31).isoformat()
            g = (1 + rng.normal(0.04 + 0.18 * quality, 0.10)) ** (-yr_back)
            revenue = rev0 * g
            gross = revenue * gm_pct
            ebitda = revenue * ebitda_pct
            dep = revenue * np.clip(rng.normal(0.035, 0.012), 0.004, 0.14)
            ebit = ebitda - dep
            interest = debt * np.clip(rng.normal(0.09, 0.015), 0.05, 0.16)
            pretax = ebit - interest
            pat_y = pretax * (1 - 0.25)
            cfo = pat_y * conv + dep
            capex = -revenue * np.clip(rng.normal(0.05, 0.03), 0.002, 0.20)
            stmt_rows.append({
                "symbol": sym, "period": period,
                "revenue": revenue, "cogs": revenue - gross, "gross_profit": gross,
                "ebitda": ebitda, "ebit": ebit, "interest": interest,
                "pretax": pretax, "pat": pat_y,
                "equity": equity0 * (0.82 ** yr_back),
                "total_debt": debt, "long_term_debt": debt * 0.7,
                "total_assets": (equity0 + debt) * 1.25,
                "cfo": cfo, "capex": capex,
            })

        # --- corporate announcements ----------------------------------------
        pos = ["Receipt of new order worth Rs 480 crore",
               "Board approves capacity expansion and capex plan",
               "Credit rating upgrade by CRISIL",
               "Announcement of buyback of equity shares",
               "Commissioning of new manufacturing plant",
               "USFDA approval received for key product",
               "Signing of MoU for strategic partnership",
               "Board recommends dividend"]
        neg = ["Resignation of Chief Financial Officer",
               "Rating downgrade by ICRA",
               "SEBI show cause notice received",
               "Invocation of pledge by lender",
               "Plant shutdown due to fire incident",
               "Auditor qualification in limited review report",
               "Promoter stake sale in open market"]
        n_ann = rng.poisson(4 + 3 * quality)
        for _ in range(n_ann):
            offs = int(rng.integers(0, 58))
            when = end - timedelta(days=offs)
            good = rng.random() < (0.35 + 0.45 * quality)
            subject = str(rng.choice(pos if good else neg))
            ann_rows.append({"symbol": sym, "date": when.isoformat(),
                             "subject": subject, "detail": ""})
        # a results filing for most names
        if rng.random() < 0.8:
            when = end - timedelta(days=int(rng.integers(5, 70)))
            ann_rows.append({"symbol": sym, "date": when.isoformat(),
                             "subject": "Outcome of Board Meeting - Unaudited Financial "
                                        "Results for the quarter ended",
                             "detail": "financial results"})

        # --- bulk deals -------------------------------------------------------
        if rng.random() < 0.25:
            when = end - timedelta(days=int(rng.integers(0, 40)))
            qty = float(volume[-1] * rng.uniform(0.3, 3.0))
            deal_rows.append({"symbol": sym, "date": when.isoformat(),
                              "client": "SIMULATED FUND",
                              "side": "BUY" if rng.random() < (0.3 + 0.5 * quality) else "SELL",
                              "qty": qty, "price": float(close[-1]), "kind": "BULK"})

    prices = pd.concat(price_rows, ignore_index=True)
    store.upsert_prices(prices)

    # --- indices ------------------------------------------------------------
    piv = prices.pivot_table(index="date", columns="symbol", values="close")
    eq = (1 + piv.pct_change().mean(axis=1).fillna(0)).cumprod() * 22000
    idx_rows = [pd.DataFrame({"index_name": "NIFTY 50", "date": eq.index,
                              "open": eq.values, "high": eq.values * 1.004,
                              "low": eq.values * 0.996, "close": eq.values,
                              "pct_change": eq.pct_change().fillna(0).values * 100})]
    vix_series = 11 + 40 * pd.Series(eq).pct_change().rolling(20).std().fillna(0.008).values
    idx_rows.append(pd.DataFrame({"index_name": "INDIA VIX", "date": eq.index,
                                  "open": vix_series, "high": vix_series,
                                  "low": vix_series, "close": vix_series,
                                  "pct_change": 0.0}))
    store.upsert_indices(pd.concat(idx_rows, ignore_index=True))

    # --- fundamentals -------------------------------------------------------
    today = date.today().isoformat()
    for sym, payload in fund_rows:
        store.upsert_fundamentals(sym, today, payload)

    # --- announcements (scored through the real lexicon) --------------------
    from .ingest import score_announcement
    a = pd.DataFrame(ann_rows)
    if not a.empty:
        scored = a.apply(lambda r: score_announcement(f"{r['subject']} {r['detail']}"),
                         axis=1, result_type="expand")
        a["score"], a["tags"] = scored[0], scored[1]
        store.upsert_announcements(a)

    if deal_rows:
        store.upsert_deals(pd.DataFrame(deal_rows))

    # --- corporate actions, derivatives, shareholding -----------------------
    if ca_rows:
        store.upsert_corporate_actions(pd.DataFrame(ca_rows))
    if fno_rows:
        store.upsert_fno(pd.concat(fno_rows, ignore_index=True))
    if sh_rows:
        store.upsert_shareholding(pd.DataFrame(sh_rows))
    if stmt_rows:
        sdf = pd.DataFrame(stmt_rows)
        for sym, g in sdf.groupby("symbol", sort=False):
            wide = g.drop(columns=["symbol"]).set_index("period")
            store.upsert_statements(sym, wide, fetched=today)

    # --- market-wide flows (FII/DII cash + FII index futures positioning) ----
    mf_rows = []
    recent = eq.index[-45:]
    fii_bias = rng.normal(0, 1)
    for i, dt in enumerate(recent):
        fii_net = float(rng.normal(400 * fii_bias, 2200))
        dii_net = float(rng.normal(-0.55 * fii_net + 300, 1400))
        mf_rows.append({"date": dt, "metric": "FII_net_cr", "value": fii_net})
        mf_rows.append({"date": dt, "metric": "DII_net_cr", "value": dii_net})
        mf_rows.append({"date": dt, "metric": "FII_index_fut_long_pct",
                        "value": float(np.clip(45 + 12 * fii_bias + rng.normal(0, 5), 5, 95))})
    store.upsert_market_flows(pd.DataFrame(mf_rows))

    # --- security master ----------------------------------------------------
    sec_master = pd.DataFrame({
        "symbol": syms,
        "name": [s.title() + " Limited" for s in syms],
        "series": "EQ", "surveillance": "", "updated": today,
    })
    store.upsert_securities(sec_master)
    store.set_meta("simulated", True)
