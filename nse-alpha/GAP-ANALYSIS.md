# NSE Alpha vs. the Master Guide — coverage audit

Audit date: 19 Aug 2026. Every claim below was checked against the code, not
recalled. Section numbers refer to *Indian Stock Market & Financial Analysis —
Complete Reference Manual*.

> **Update, same day.** The metrics gap has been closed — `nsealpha/financials.py`
> now pulls annual income statement, balance sheet and cash flow per company, and
> derives interest coverage, free cash flow, cash conversion, gross margin, real
> ROCE and multi-year ROE/ROCE consistency. Rows marked ✅ **(new)** below were
> added in that pass. The three structural gaps — the checklist gate, capital
> gains tax, and the missing sell side — are still open.

---

## The headline finding: gate vs. rank

This is more important than any single missing metric.

The guide's §7 is a **hard gate**: "MANDATORY BUY CHECKLIST (must fulfil ≥ 5
criteria)". A stock with D/E of 3.0, or an auditor qualification, or promoter
pledge at 40%, is **disqualified** — it does not appear, no matter how good the
chart looks.

NSE Alpha is a **continuous ranking**. Those same problems become bad z-scores on
individual sub-factors, which can be outvoted by strength elsewhere. A stock with
terrible leverage but spectacular momentum, cheap valuation and a long build-up in
futures OI can still reach the top 10.

Both designs are defensible — a gate is safer, a ranking finds more opportunities —
but they are **not the same thing**, and the current system does not implement the
guide's stated philosophy. Only the pre-scoring filters in `universe.py` (series,
price, liquidity, delivery %, surveillance) behave as true gates today.

**Fix:** add an optional `checklist` mode that applies §7 as a hard pre-filter
before ranking, and shows a per-stock "5 of 6 criteria met" badge. Roughly a day's
work, and it makes the two systems agree.

---

## §2 — P&L flow (Revenue → COGS → … → PAT)

| Concept | Status | Notes |
|---|---|---|
| Revenue / top-line growth | ✅ | `revenue_growth` in the fundamental block |
| COGS | ✅ **(new)** | Read from the income statement; used to derive gross profit where it isn't reported directly |
| **Gross profit & gross margin** | ✅ **(new)** | `gross_margin` from the income statement, scored inside `quality` |
| EBITDA | ✅ **(new)** | EBITDA and `ebitda_margin` now computed from statements |
| D&A | ❌ | Not sourced |
| EBIT / operating margin | ✅ | `operating_margin` |
| EBT / PBT | ✅ **(new)** | `pretax` stored; used in the ICR chain |
| PAT / net margin | ✅ | `profit_margin`, `pat_growth_yoy`, `pat_growth_qoq` |

---

## §3 — Valuation & capital efficiency ratios

| Ratio | Guide target | Status | Notes |
|---|---|---|---|
| EPS | — | 🟡 | EPS *growth* used; EPS level never used |
| P/E vs peers | — | ✅ | Scored against the stock's own sector P/E — the strongest part of our valuation block |
| **P/E vs own 5-yr history** | at/below 5-yr median | ❌ | Guide asks for both peer *and* historical comparison. We do peer only |
| PEG | < 1.0 undervalued | 🟡 | Implemented; our curve (good 0.6 / bad 3.0) is stricter than the checklist's ≤ 1.5 |
| P/B | — | ✅ | |
| Enterprise Value | — | 🟡 | Only as EV/EBITDA; EV itself never computed |
| EV/EBITDA | — | ✅ | |
| ROE | > 15–18% | ✅ | Scored bad 6 / good 25, so 15–18 sits mid-scale |
| ROCE | > 15–20% | ✅ **(new)** | Now the real thing: EBIT / (equity + long-term debt). The ROA × 1.4 stand-in remains only as a last resort and is flagged via `_roce_estimated` |
| D/E | < 0.5 | ✅ | Continuous, not a gate |
| **Interest Coverage Ratio** | > 4.0× | ✅ **(new)** | EBIT / \|interest\|. Scored in `quality`, and a red flag fires below the guide's 4.0× floor. Debt-free companies score maximum, not missing |
| **Free Cash Flow (CFO − Capex)** | 5-yr cumulative positive | ✅ **(new)** | Per-year FCF, cumulative total, and count of positive years, in the new `cash_quality` sub-block |

---

## §4 — Market operations

| Item | Status | Notes |
|---|---|---|
| Trading sessions / timings | n/a | Irrelevant to an end-of-day screen |
| T+1 / T+0 settlement | n/a | Backtest is close-to-close, so implicitly consistent |
| Short delivery & auction | n/a | |
| CNC / MIS / limit / market / SL-L / SL-M / GTT | 🟡 | We emit a stop and targets but never say **which order type** to place. Easy, useful addition: "CNC, with an SL-M at ₹969 and a GTT at ₹1,141" |
| **Individual scrip circuit filters (±2/5/10/20%)** | ❌ | Never implemented, despite being flagged in the original design. Frequent circuit-hitting is a strong illiquidity/manipulation signal and should be a universe filter |
| Index circuit breakers | n/a | Market-halt mechanics, not a ranking input |

---

## §5 — Taxation, costs & corporate actions

| Item | Status | Notes |
|---|---|---|
| **STCG 20% (≤ 12 months)** | ❌ | **Not modelled anywhere** |
| **LTCG 12.5% (> 12 months), ₹1.25 L exempt** | ❌ | **Not modelled anywhere** |
| Intraday / F&O taxed at slab | n/a | We don't recommend intraday or F&O trades |
| STT, stamp duty, GST, exchange & SEBI fees | 🟡 | Lumped into one `cost_bps: 35` in the backtest; never itemised |
| Stock split | ✅ | `adjust.py` |
| Bonus shares | ✅ | `adjust.py` |
| Record date vs ex-date | ✅ | `adjust.py` uses ex-date correctly |

**Why the tax gap matters more than it looks.** The backtest default is a **5-day
hold**. Every gain is therefore STCG at 20%, but the backtest charges 0% tax. That
overstates net returns by a fifth of every winning trade — and it silently biases
the whole design toward short holding periods, because the tax penalty for churning
is invisible. A 5-day-hold and a 13-month-hold strategy are not comparable on the
numbers currently reported.

---

## §6 — End-to-end analysis framework

### Step 1: fundamental & moat (what to buy)

| Item | Status | Notes |
|---|---|---|
| Competitive moat | ⬜ | Genuinely not automatable. Should be stated as out of scope, not quietly omitted |
| Industry runway / TAM ≥ 12–15% | ❌ | No data source in the project |
| Promoter pledge < 5% | 🟡 | Tracked; our red flag fires at **> 10%**, the guide says **< 5%** |
| Related-party transactions | ⬜ | Not automatable from our sources |
| Promoter ownership > 50% (mid/small) | 🟡 | `promoter_holding` scored continuously (bad 25 / good 62); no explicit > 50% test |
| **Cash-flow conversion: 5-yr CFO ≥ PAT** | ✅ **(new)** | `cash_conversion` = cumulative CFO / cumulative PAT, scored and flagged below 0.8× |

### Step 2: technical timing (when to buy)

| Item | Status | Notes |
|---|---|---|
| Above rising 200 DMA | ✅ | `above200` + `sma200_rising` in `trend_structure` |
| Breakout volume ≥ 1.5–2.0× 20-day avg | ✅ | `vol_surge`, curve caps at 2.0× |
| **Weekly RSI > 50** | ❌ | We compute **daily** RSI only. Weekly RSI is a different, slower signal |
| Avoid daily RSI > 80 | 🟡 | Our penalty band starts at **82**, not 80 |

### Step 3: post-purchase tracking

| Item | Status | Notes |
|---|---|---|
| Quarterly earnings scorecard | 🟡 | Revenue growth and operating margin tracked; no quarter-on-quarter *trend*, no guidance/concall data |
| Alpha vs sector index; review if underperforming > 18 months | 🟡 | Mansfield RS vs benchmark exists; the **18-month review trigger** does not, and RS is measured against the broad index, not the stock's **sector** index |
| FII/DII quarterly ownership shifts | ✅ | The flows block |

---

## §7 — Buy checklist and sell triggers

### Mandatory buy checklist (≥ 5 of 6)

| # | Criterion | Status |
|---|---|---|
| 1 | ROCE > 15% **and** ROE > 15%, **3–5 years running** | ✅ **(new)** `roce_consistency` and `roe_consistency` measure the share of available years above 15%, scored in `quality` |
| 2 | D/E < 0.5 **and ICR > 4.0×** | ✅ **(new)** Both now present |
| 3 | Cumulative 5-yr FCF positive | ✅ **(new)** |
| 4 | PEG ≤ 1.5 **or** at/below 5-yr median P/E | 🟡 PEG yes, historical P/E missing |
| 5 | No auditor qualifications, no sudden CFO exits, pledge < 5% | 🟡 All three are in the announcement lexicon and pledge data — but as **soft negative scores**, not disqualifiers |
| 6 | Above 200 DMA with stage-2 accumulation volume | ✅ |

### Disciplined sell triggers

| # | Trigger | Status |
|---|---|---|
| 1 | Thesis invalidation | ⬜ Not automatable |
| 2 | **Operating margin compression, 3 consecutive quarters** | ❌ No quarter-over-quarter margin trend is stored |
| 3 | Governance red flags / **rising promoter pledge** | 🟡 Rising pledge is flagged; related-party and promoter loans are not automatable |
| 4 | Extreme valuation vs forward earnings | 🟡 Partially, via PEG |
| 5 | Stop-loss violated: **7–8% for swing**, **below 200-week MA** for long-term | 🟠 Our stop is ATR-based, capped at 12%. **200-week MA is not computed at all** |

**The bigger point: there is no sell side.** The project has no concept of a
position you already hold. Every §7 sell trigger and all of §6 Step 3 assume you
own something and are deciding whether to keep it. NSE Alpha only ever answers
"what should I buy today". That is half the guide's workflow, and it is the
largest single gap in the project.

---

## Summary

| | Before | After the statements pass |
|---|---|---|
| ✅ Fully covered | 17 | **27** |
| 🟡 Partially covered / calibrated differently | 16 | 15 |
| 🟠 Covered by approximation | 3 | 0 |
| ❌ Not covered, and implementable | 13 | 4 |
| ⬜ Not automatable (correctly out of scope) | 4 | 4 |

Still ❌: capital-gains tax in the backtest, weekly RSI, the 200-week MA, and
circuit-filter detection. Plus the two structural items — the checklist gate and
the sell side — which are design decisions rather than missing metrics.

### Ranked by what I'd fix first

1. **Capital-gains tax in the backtest** — the reported numbers are currently
   wrong in a direction that flatters short holding periods.
2. **A portfolio / sell-side module** — half the guide's workflow is missing.
3. **The checklist gate** — make the system implement the philosophy it claims.
4. **ICR, FCF and CFO/PAT conversion** — three named checklist metrics, absent.
   All three need a cash-flow and balance-sheet source (NSE XBRL filings, or
   yfinance's cashflow statements).
5. **Weekly RSI and the 200-week MA** — cheap; both are two lines given the data
   we already hold.
6. **Circuit-filter detection** — a real manipulation signal, cheap to add from
   bhavcopy data we already download.
7. **Multi-year ROE/ROCE consistency and 5-yr median P/E** — needs a fundamentals
   history table, which we don't keep (this is the same root cause as the
   point-in-time backtest bias already documented in the README).
8. **Threshold alignment** — RSI 80 not 82, pledge 5% not 10%, swing stop 7–8%.
   Trivial, but they should match the document they claim to follow.
