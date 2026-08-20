# NSE Alpha

A multi-factor daily ranking system for Indian equities. It pulls every NSE-listed
stock from the exchange's own published files, filters out what you couldn't
realistically trade, scores what's left on six independent analytical blocks, and
publishes a website — and optionally an evening email — with the ten highest-ranked
names, why each one ranked, what could go wrong with it, and where the stop-loss and
position size would sit.

**This is a screen, not advice.** It narrows two thousand stocks to ten worth
researching. It cannot read an annual report, smell an accounting fraud, or know
that the promoter is under investigation. Read the disclaimer at the bottom.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Check everything works, with a synthetic market. No network needed.
python run_daily.py --simulate --serve

# 2. First real run — downloads ~2 years of NSE data. Takes 20-60 minutes.
python run_daily.py --refresh --backfill

# 3. Every trading evening after that (after ~6:30 pm IST). Takes 2-5 minutes.
python run_daily.py --refresh --serve

# 4. Or explore it interactively.
streamlit run app.py
```

The site lands in `site/index.html`. Open it directly, or serve the folder:

```bash
python -m http.server 8000 --directory site   # then visit localhost:8000
```

Each run also archives a dated copy in `site/archive/`, so you can go back and
check what the model said three weeks ago against what actually happened. That
habit is worth more than any single day's list.

---

## What it actually does

### 1. Data — straight from NSE

No scraping of third-party sites, no paid API. Everything comes from files NSE
publishes itself:

| What | Source |
|---|---|
| Daily OHLC + volume + **delivery %** for every equity | `sec_bhavdata_full_DDMMYYYY.csv` |
| Fallback bhavcopy | UDiFF `BhavCopy_NSE_CM_..._F_0000.csv.zip` |
| Index closes (benchmark, sectors, India VIX) | `ind_close_all_DDMMYYYY.csv` |
| Surveillance flags (ASM/GSM) | `sec_list_DDMMYYYY.csv` |
| Corporate filings | `/api/corporate-announcements` |
| Splits, bonuses, rights (ex-dates) | `/api/corporates-corporateActions` |
| Bulk & block deals | `/api/historicalOR/bulk-block-short-deals` |
| F&O open interest, PCR, basis | `BhavCopy_NSE_FO_..._F_0000.csv.zip` |
| Participant-wise OI (FII/DII/pro/client) | `content/nsccl/fao_participant_oi_DDMMYYYY.csv` |
| FII/DII daily cash activity | `/api/fiidiiTradeReact` |
| Quarterly shareholding pattern | `/api/corporate-share-holdings-master` |
| Annual income statement / balance sheet / cash flow | yfinance statements |
| P/E, sector P/E, industry | `/api/quote-equity` |
| Ratios NSE doesn't publish (ROE, D/E, P/B, margins) | yfinance, optional |

The key architectural choice: **history is built from bhavcopies, not per-symbol
calls.** One bhavcopy is the entire market for one day. Two years costs ~500
requests instead of ~2,000, and it's the only practical way to cover every listed
name without being rate-limited into oblivion. Everything lands in one SQLite file
(`data/market.sqlite`) and is cached forever — historical bhavcopies never change,
so a daily run downloads exactly one new file.

If NSE starts returning 403s (it does this to datacentre IPs), set
`data.server_mode: true` in `config.yaml` — that switches to HTTP/2 via httpx,
which NSE's edge is more relaxed about.

### 1a. Corporate actions — the bug that silently ruins screeners

NSE bhavcopy reports **raw traded prices**. It does not adjust history for splits
or bonuses. So a 1:5 split appears as

```
Friday close   2,480
Monday close     498          <- a "-80% day", as far as any factor can tell
```

and that single fake day corrupts every momentum horizon, the 52-week high and
your distance from it, ATR (and therefore every stop-loss the tool suggests),
realised volatility, OBV, the Hurst exponent, and the historical-edge factor.
Unfixed, the screen systematically punishes companies that just rewarded their
shareholders — and it does it silently, which is the dangerous part.

`nsealpha/adjust.py` fixes this with two independent sources, cross-checked:

1. **PREV_CLOSE (primary).** On an ex-date NSE populates the bhavcopy's
   `PREV_CLOSE` with the *adjusted* previous close — it has to, because that is
   what the exchange uses to compute the day's circuit limits. So
   `prev_close(t) / close(t-1)` is 1.0 on a normal day and ~0.2 on a 1:5 split.
   The adjustment factor falls out of data you already have, with no extra
   request and no free-text parsing.
2. **The corporate-actions feed (corroboration).** Ex-date plus a purpose string
   like `FACE VALUE SPLIT FROM RS 10/- TO RS 2/-` or `BONUS 1:1`, parsed to a
   ratio and compared against the implied factor.

Where they agree, confidence is high. Where they disagree, the stated filing wins
and the mismatch is **logged and surfaced in the site's data-quality panel** rather
than quietly resolved — a disagreement usually means one feed is wrong and you
want to know which stock is affected.

Adjustment runs backwards: today's price is never touched, so the entry, stop and
target levels printed are always the real numbers on your broker's terminal.

There is a test (`tests/test_parsers.py::test_corporate_action_adjustment`) that
injects a 1:5 split exactly the way NSE reports one and asserts the raw 12-month
return reads −76% while the adjusted one reads +18%. That is the whole bug and
the whole fix in one assertion.

### 2. Filters — the unglamorous part that matters most

Ranking every listed security would fill the top 10 with ₹14 stocks that went up
400% on 9% delivery. Those look spectacular on momentum and are untradeable. So a
stock is dropped, before any scoring, if it fails **any** of:

- series other than `EQ` (BE/BZ = trade-to-trade or surveillance)
- price below ₹20
- 20-day average turnover below ₹5 crore
- fewer than 250 days of price history
- didn't trade on 18 of the last 20 sessions
- 60-day average delivery below 20% (persistent intraday churn)
- flagged on NSE's ASM/GSM surveillance lists
- not a common equity (ETF, SGB, REIT, InvIT, government security)

Every threshold is in `config.yaml`. The site shows you how many stocks each rule
removed, so you can see what you're excluding.

### 3. The six factor blocks

Each block produces sub-factors oriented so higher = better. Those are winsorised
at the 2nd/98th percentile, z-scored across the universe (optionally *within
sector*, which stops the list becoming ten PSU banks in a month when PSU banks are
ripping), blended within the block, then blended across blocks.

**Technical — 22%.** Moving-average structure (close > SMA20 > SMA50 > SMA200 with
a rising 200-DMA is a full-marks stage-2 advance; a death cross subtracts), ADX and
directional index for trend *strength*, an RSI zone score that is deliberately
non-monotonic (55–70 scores best, above 80 is penalised as overbought, below 35 as
a falling knife), MACD histogram sign and slope, Bollinger %B plus a squeeze
indicator, Supertrend direction and its ATR buffer, OBV slope and volume surge, 52-
week-high proximity and 20-day breakout, and **delivery-percentage trend** — the
NSE-specific one. Rising delivery alongside rising price means buyers are taking
shares home rather than churning intraday.

**Momentum — 18%.** 1/3/6-month returns, plus 12-month momentum *excluding the last
month* (the academic standard — the most recent month tends to short-term reverse,
so including it dilutes the signal), Mansfield relative strength versus the
benchmark and whether that RS line is still rising, and a consistency score:
percentage of positive weeks, return/volatility, and how much of the move came from
a single gap. A stock that ground out +40% over six months is a better hold than
one that gapped +40% in a week.

**Fundamentals — 22%.** Five sub-blocks.

*Valuation* is led by **P/E relative to the stock's own sector P/E** — an absolute
P/E of 40 means nothing without knowing whether the sector trades at 15 or 60 —
plus P/B, PEG, EV/EBITDA and dividend yield, with a value-trap discount so absurdly
cheap stops earning marks.

*Quality* is ROE, **real ROCE** (EBIT over capital employed, computed from the
balance sheet rather than approximated), **interest coverage** (EBIT / interest,
against a 4.0× floor), gross and operating margin, debt-to-equity (weighted hard,
because Indian mid-caps die of leverage), current ratio, and **multi-year
consistency** — the share of the last five years in which ROE and ROCE cleared 15%,
because one good year is not a track record.

*Growth* is YoY revenue, PAT and EPS growth plus QoQ, haircut when it's being
bought with debt.

*Cash quality* is **free cash flow** (operating cash flow minus capex), how many of
the last five years it was positive, and the **cash-conversion ratio** — cumulative
CFO over cumulative PAT. This gets its own sub-block rather than being averaged in,
because reported profit is an opinion and cash is a fact. The classic accounting
failure in Indian small and mid caps is profit that grows beautifully while
operating cash flow does not follow: receivables balloon, inventory piles up, and
the profit turns out to be a ledger entry. Five years of CFO ≥ PAT is very hard to
fake, and a strong ROE would otherwise hide its absence.

*Ownership* is promoter holding, promoter pledge, and institutional holding.

Banks and NBFCs are excluded from interest coverage, gross margin and ROCE —
interest is their raw material, not a financing cost, so those numbers would be
nonsense. They still get ROE, cash conversion and everything else.

**News — 8%.** NSE corporate filings scored through a keyword lexicon
(order wins, capex, rating upgrades, buybacks, USFDA approvals positive; auditor
qualifications, defaults, pledge invocations, SEBI notices, promoter selling
negative), decayed with a 10-day half-life because a three-week-old order win is
already in the price. A single severe negative isn't averaged away by ten routine
positives. Plus bulk/block deal direction, and how the stock behaved in the three
sessions after its last results filing relative to the index — the market's verdict
on the numbers beats our reading of them.

Deliberately **no news scraping or social sentiment**. On Indian small and mid caps
that channel is dominated by paid promotion and Telegram pump groups; feeding it
into a ranking model is how you end up long exactly the stocks the operators want
you long.

**Past trends — 18%.** The block that asks a different question: *when this
particular stock has looked like this before, what happened next?* For every past
day we compute the same lightweight setup score, find the historical days in the
same bucket as today, and average the forward 10-day return from those days — a
per-stock conditional expectation, with a confidence discount for thin samples. A
chart that looks perfect but has repeatedly failed *in this name* gets marked down.
Alongside it: the Hurst exponent (a trend-following signal on a mean-reverting
stock is a losing trade in a nice suit), month-of-year seasonality, drawdown
recovery record, and whether volatility is expanding into the signal.

**Institutional flows — 12%.** What large money is *doing*, as opposed to what the
chart looks like. Four things:

*Futures open-interest build-up* is the one worth understanding. Price up with OI
up is a **long build-up** — new money is backing the move, and it scores best.
Price up with OI *down* is **short covering**: the move is people closing losing
bets, not anyone believing in the stock, and such rallies typically stall. On a
price chart those two are indistinguishable; on OI they are opposites, and the
model treats them as such.

*Put-call ratio* on open interest, read only at extremes — a high PCR means put
writers (usually the informed side in Indian options) are comfortable selling
downside. The middle of the range is scored flat on purpose, because it carries
no information.

*Futures basis*: a premium means leveraged money is paying to be long; a persistent
discount on a rising stock says the rally isn't supported by derivatives
positioning, which is the classic setup for a fast unwind. That one also fires a
red flag.

*Institutional ownership*: quarter-on-quarter change in FII, DII and promoter
holding, plus any change in pledged shares, from the shareholding pattern. Slow-
moving, but it's the only ownership signal that covers small and mid caps.

Only ~200 NSE stocks have listed derivatives, so the first three simply don't exist
for most of the universe. Those stocks score as **missing**, not average — the
weights redistribute across blocks that do have data. A stock without futures isn't
"neutrally positioned"; it's unmeasurable on that axis, and treating missing as
neutral would quietly bias the whole ranking toward large caps.

Market-level flows — FII/DII daily cash activity and participant-wise open interest
— feed the regime overlay rather than individual scores, which is where they belong.

**What's deliberately not here:** AMFI monthly mutual fund portfolio disclosures.
They'd be excellent — they show exactly what each fund bought last month — but
they're published per-AMC in inconsistent Excel layouts across forty-odd asset
managers with no common schema. That's a scraping project of its own and it would
break constantly. The quarterly shareholding pattern gives a coarser version of the
same signal from one clean official source.

### 4. Risk overlay — when *not* to buy

A ranking model always produces a top 10, including on the morning of a crash.
Ranking is relative; survival is absolute. So before any pick is shown, the market
itself is assessed: benchmark versus its 200-DMA and 50-DMA, breadth (what share of
the universe is above its own 50-DMA), India VIX, FII/DII net flows over the last
20 sessions, and how long FIIs are positioned in index futures. That yields
**RISK-ON / CAUTION / RISK-OFF** and a suggested exposure multiplier
(100% / 60% / 25%). The picks still appear in risk-off — with the warning attached
and smaller sizes.

The FII/DII pair is read together rather than separately: heavy FII selling that
DIIs fully absorb is a very different market from heavy FII selling into no bid,
and the overlay says which one you're in.

Then per pick: an ATR-based stop (sized to the stock's own normal daily travel,
slid below a nearby SMA when one sits just underneath, capped at 12%), targets at
2R and 3.5R, and a share count such that a stop-out costs exactly 1% of your
capital. A sector cap (max 3) stops the list being one bet ten times.

Finally, red flags are surfaced per pick even when it ranks first — RSI above 78,
ATR above 6%, thin liquidity, promoter pledge, high leverage, a recent negative
filing, volume 3.5× normal.

---

## The Streamlit app

```bash
streamlit run app.py          # opens at localhost:8501
```

Three outputs, three jobs. The **static site** is a report — a fixed record of what
the model said on a given evening, archived and emailable. The **email** is the
push notification. The **Streamlit app** is a workbench: it exists to answer *what
if*, which neither of the other two can.

Five tabs:

**Top picks** — the same cards as the site, but live. Regime banner, factor
breakdown, trade plan, red flags, plus where each pick sits in the distribution of
all scores and which sectors are strongest today.

**Explore** — the full ranking, sortable and filterable by symbol, sector,
liquidity and RSI. Download any view as CSV.

**Stock deep dive** — any stock in the universe, not just the picks. Candlesticks
with SMAs, Bollinger and Supertrend, your stop and targets drawn on the chart, then
volume, delivery %, RSI and MACD in stacked panels. Below that, every sub-factor
z-score and its exact contribution to the composite, so you can see precisely why
a stock ranks 4th and not 40th.

**Backtest** — run it with different hold periods, costs and start dates without
editing config.yaml.

**Email & schedule** — configure SMTP (written to `.env`, chmod 0600), send
today's digest immediately, preview it without sending, and turn the daily
schedule on or off. See [Scheduling it](#scheduling-it--the-easy-way) below for
what that button actually installs.

**Data quality** — every corporate action applied, which of them the two sources
disagreed on, factor coverage per block, what got filtered out and why, and the
FII/DII flow chart.

### The weight sliders

This is the point of the app. The six factor weights are sliders in the sidebar;
move one and the entire market re-ranks in about half a second. It's the fastest
way to build intuition for what the model is actually doing — turn Momentum up and
watch the value names fall out; zero out Fundamentals and see which picks were
riding on cheapness alone.

That speed comes from splitting the pipeline in two (see `pipeline.py`):

| | what it does | cost | cached? |
|---|---|---|---|
| `prepare()` | load prices, adjust for corporate actions, compute ~25 indicators per stock, run the six factor blocks | ~30s (2000 stocks: a few minutes) | yes |
| `score_from_prepared()` | winsorise, z-score, blend, rank, size positions | ~450ms | no — runs on every slider move |

Everything expensive is weight-*independent*, so it's computed once. Without that
split each slider nudge would trigger a full recompute and the app would be
unusable. The CLI and the backtest call both halves back to back, so nothing else
changed.

Sliders never write to `config.yaml` — your cron job keeps using the saved weights.
When you find a combination you like, edit the file to make it permanent.

### Notes

- **Run it locally.** It needs the SQLite database, which is built from NSE data on
  your machine and runs to tens of megabytes. Streamlit Community Cloud has no
  route to nseindia.com and no database to read.
- The **"Score as of"** date picker rolls the model back in time — useful for
  checking what it would have said last Tuesday against what actually happened.
- The app is read-only. Data refresh stays with `run_daily.py --refresh`, so a
  browser tab can't accidentally kick off a 40-minute download.

---

## The daily email

```bash
cp .env.example .env          # fill in your SMTP details
python run_daily.py --simulate --email-preview    # check the layout, sends nothing
python run_daily.py --refresh --email             # the real thing
```

`--email-preview` writes `site/digest_preview.html` and `site/digest_preview.txt`
so you can look at both parts before wiring up cron.

The email is multipart: a styled HTML version, and a plain-text version that is
written properly rather than as an afterthought — that's what you actually see in
a lock-screen notification. It leads with **what changed** (which names are new,
which dropped out, how far each one moved) rather than making you re-read ten rows
from scratch every evening, because that is how people stop opening the email in
week three. Each pick carries entry zone, stop, targets, quantity, the top three
reasons it ranked, and its red flags.

### Gmail setup

Your normal Google password will be rejected by SMTP. You need an **App Password**:

1. Google Account → Security → turn on 2-Step Verification (required)
2. Security → App passwords → select "Mail" → Generate
3. Paste the 16-character code into `SMTP_PASSWORD` in `.env` (spaces are fine)

Outlook (`smtp-mail.outlook.com:587`), Zoho (`smtp.zoho.in:587`) and any other SMTP
server work the same way; port 465 switches to implicit TLS automatically.

### Scheduling it — the easy way

Open the app, go to **Email & schedule**, fill in your SMTP settings, and press
**Turn on daily email**. Pick the hour and whether it should skip weekends.

That button does *not* run the schedule inside Streamlit. Streamlit only executes
Python while a browser tab is open, so a toggle that lived in the app would
silently stop working the moment you closed it — the worst possible failure mode
for something whose whole job is to be reliable at 7pm. Instead it installs an
entry in your **operating system's** scheduler (cron on Linux/macOS, Task
Scheduler on Windows), which keeps running with the app closed and the browser
quit.

The cron entry is wrapped in marker comments and **only that block is ever
touched** — `crontab` has no edit-one-line operation, so the whole file is
rewritten, and the merge that does it is a pure function with tests asserting that
your other jobs, comments, blank lines and `MAILTO=` survive byte-for-byte.
"Turn off" removes only our block.

One caveat: the job runs on *that machine*. If it's asleep or switched off at the
scheduled time, cron does not catch up afterwards — you simply get no email that
evening. A laptop that's shut by 7pm is a poor host; an always-on desktop or a
small VPS is a good one.

### Scheduling it — by hand

`schedule_daily.sh` already runs `--refresh --email`. Add one cron line:

```
0 19 * * 1-5  /full/path/to/nse-alpha/schedule_daily.sh
```

7:00 pm IST, Monday to Friday — after NSE publishes the full bhavcopy. On trading
holidays the bhavcopy 404s, the pipeline reuses the last session, and nothing
breaks. The script also writes a loud line into the log if a run failed to produce
a site, so a silent cron failure doesn't look the same as a quiet market day.

The app's **Email & schedule** tab reads the same logs and will tell you if the
last scheduled run didn't finish, whichever way you set it up.

**Note:** the email has to be sent from wherever the pipeline runs, i.e. your own
machine or a small VPS. A cloud scheduler that can't reach nseindia.com can't
produce the picks in the first place.

---

## Backtesting

```bash
python run_daily.py --backtest
```

Walk-forward: step through history every 5 trading days, re-run the **same
`pipeline.run`** the live screen uses with `as_of` set to that date so nothing sees
the future, buy the top 10 equally weighted, hold to the next step, charge 35bps
round trip, compare to the benchmark. Results land in `site/backtest_*.csv` and a
summary is embedded in the site.

Three honest caveats, which matter more than the headline number:

1. **Fundamentals are a current snapshot, not point-in-time.** NSE publishes no
   clean historical ratio archive, so the fundamental block carries look-ahead bias
   in a backtest. For a clean read, set `weights.fundamental: 0`, re-run, and
   compare the two.
2. **Announcements only reach back as far as you've ingested them.**
3. **Survivorship is better than most retail backtests** — the bhavcopy archive
   contains delisted names on the days they traded — but a stock delisted before
   your first download is simply absent.

Use the backtest to sanity-check that the weights aren't actively harmful, not to
convince yourself of an edge. If it shows a spectacular Sharpe, you have a bug.

---

## Tuning it

Everything lives in `config.yaml`; nothing is hardcoded in the scoring path.

| Want to | Change |
|---|---|
| Trade shorter-term swings | Raise `technical` and `momentum` weights, drop `fundamental` toward 0.10, set `backtest.rebalance_days: 3` |
| Invest, not trade | `fundamental: 0.45`, `technical: 0.15`, `rebalance_days: 21` |
| Stay in large caps only | `universe.source: "NIFTY 100"`, `min_avg_turnover_cr: 25` |
| Hunt small caps (riskier) | `min_avg_turnover_cr: 2`, `min_price: 10`, and raise `max_per_sector` |
| Stop clustering into hot sectors | `sector_neutral: true` (already the default) |
| Size positions off a real account | `python run_daily.py --refresh --capital 500000` |
| Weight derivatives positioning higher | Raise `flows` weight; only affects F&O names |
| Ignore F&O data entirely | `weights.flows: 0`, or `data.fetch_flows: false` to skip the download |
| Turn off corporate-action adjustment | `data.adjust_corporate_actions: false` — **don't**, unless you're debugging |

Re-run the backtest after any weight change. Changing weights until the backtest
looks good is overfitting — change them because you have a *reason*, then check the
backtest didn't fall apart.

---

## Project layout

```
run_daily.py            CLI entry point (site + email)
app.py                  Streamlit UI (streamlit run app.py)
config.yaml             every knob
.env.example            SMTP settings for the daily email (copy to .env)
schedule_daily.sh       cron wrapper for daily runs
nsealpha/
  nseclient.py          NSE session, cookies, rate limiting, retries, archives
  ingest.py             parse bhavcopies / indices / filings / corporate actions
  adjust.py             split & bonus adjustment, dual-source cross-checked
  financials.py         annual statements → ICR, FCF, cash conversion, real ROCE
  flows.py              F&O OI, PCR, basis, FII/DII, participant OI, shareholding
  store.py              SQLite persistence
  universe.py           tradability filters
  indicators.py         ~25 technical indicators, dependency-free
  panel.py              per-symbol frames + cross-sectional snapshot
  factors/
    technical.py  momentum.py  fundamental.py
    sentiment.py  trend.py     flows.py
  scoring.py            winsorise, z-score, blend, rank, explain
  risk.py               market regime, stops, sizing, sector caps, red flags
  pipeline.py           orchestration (used by both the live run and backtest)
  backtest.py           walk-forward validation
  report.py             renders the self-contained website
  charts.py             Plotly figures for the Streamlit app
  mailer.py             the daily email digest
  scheduler.py          installs/removes the cron (or schtasks) entry
  simulate.py           synthetic market for offline testing
tests/   parser, indicator, scoring and adjustment tests
data/    market.sqlite (built on first run)
cache/   raw NSE files (safe to delete, will re-download)
site/    index.html + archive/ + digest previews
```

Run the tests any time with:

```bash
python tests/test_parsers.py
```

---

## Troubleshooting

**`403` or `401` from NSE.** Their edge blocks datacentre IPs and aggressive
clients. Set `data.server_mode: true` (needs `pip install httpx[http2]`), lower
`requests_per_second` to 1.5, and run from a residential connection if you can.

**"No price data. Run with --refresh first."** The database is empty — do the
`--refresh --backfill` run once.

**Interest coverage / cash conversion show "—" everywhere.** Those come from the
annual statements, which are fetched via yfinance on a monthly cadence. Run
`--refresh` and let the statements pass finish (it's the slowest part). Note Yahoo
usually returns **four** annual periods, occasionally five — so "5-year FCF" is
really "every year available, up to five", and each metric reports the year count
alongside it. Banks and NBFCs will always show "—" for interest coverage, gross
margin and ROCE, by design.

**Fundamentals are all blank.** The per-symbol pass is the slow one and is easy to
interrupt. It refreshes weekly; run `python run_daily.py --refresh` again and let
it finish, or run with `--skip-fundamentals` and accept that block scoring as NaN
(the coverage rule will simply reweight the other blocks).

**Everything got filtered out.** Loosen `filters` in `config.yaml`, especially
`min_avg_turnover_cr` and `min_history_days`. The site lists the top rejection
reasons.

**"Turn on daily email" says scheduling isn't available.** `crontab` isn't
installed (common on minimal containers and some Docker images). Install it —
`sudo apt install cron` on Debian/Ubuntu — or use the manual cron line the app
shows you in that same panel.

**The schedule is on but no email arrives.** Check `logs/cron.log`. The most
common causes are the machine being asleep at the scheduled time (cron doesn't
catch up), or cron running with a different PATH than your shell — which is why
the installed entry uses absolute paths for both Python and the script.

**The email doesn't send / Gmail says "Username and Password not accepted".** You
used your Google account password. SMTP needs an App Password — see the Gmail
setup section above. If 2-Step Verification is off, the App passwords menu won't
appear at all.

**Every pick shows "Institutional flows: n/a".** No F&O data yet. Run
`--refresh` once more and check the log for "F&O bhavcopy" errors; the FO archive
uses a different path from the cash bhavcopy and can be blocked separately.

**The corporate-actions panel shows disagreements.** The filing ratio and the
implied factor differ for those stocks. The stated filing was used. Worth checking
those names by hand — usually it's a rights issue (which the parser deliberately
skips, since it needs the issue price) or a mangled purpose string.

**The Streamlit app is slow to start.** The first load runs `prepare()` — indicators
for the whole universe. It's cached for six hours after that, and slider moves
never re-trigger it. Changing a *filter* (not a weight) does invalidate the cache,
because filters change which stocks get computed at all.

**The app says "SQLite objects created in a thread…".** Fixed — the store now opens
with `check_same_thread=False` and serialises every method on one lock. If you see
it, you're on an older copy.

**The backtest says annualised return and Sharpe are "withheld".** You ran fewer
than 12 periods. Annualising three five-day windows produces headline numbers from
what is statistically three coin flips, so those figures are suppressed rather than
printed with a caveat nobody reads. Raise the period count.

**The picks barely change day to day.** That's correct and expected. These are
multi-week signals; a screen whose top 10 turns over completely every morning is
measuring noise. Turnover of one or two names a day is healthy.

---

## Disclaimer

This software is provided for research and educational purposes. It is not
investment advice, and its author is not a SEBI-registered investment adviser.

Every factor here is derived from historical data, and the relationship between
past and future returns is unstable by nature — a model that ranked well last year
can rank badly this year for reasons no backtest anticipated. Backtested
performance does not predict future results. The position sizes, stop levels and
targets shown are arithmetic illustrations of a fixed-fractional risk rule, not
recommendations to transact.

The model has no knowledge of pending litigation, related-party transactions,
accounting irregularities, promoter character, regulatory action in progress, or
anything else that does not appear in price, volume, exchange filings and published
ratios. Data sourced from NSE and third parties may be delayed, incomplete or
wrong.

Do your own research, consider consulting a SEBI-registered investment adviser, and
never risk money you cannot afford to lose.
