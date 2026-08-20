# BUSINESS.md — NSE Alpha, explained from scratch

> Written using the Feynman technique: every idea is explained as if to a smart
> 15-year-old who has never opened a stock trading app. No jargon is used
> without first being defined in plain English. If a sentence still sounds like
> gibberish, that's a bug in this document — the glossary at the bottom has
> every term again, alphabetically, in case you jump straight there.

---

## 1. What is this thing, in one breath?

Every evening, after the Indian stock market closes, this program looks at
**every single stock on the NSE** (the National Stock Exchange of India —
roughly 2,000 companies), throws out the ones that are too risky or too hard to
actually trade, scores the rest on six completely different tests, and hands
you a list of the **top 10** — with a reason for each one, a warning about what
could go wrong, and exact numbers for where to place your stop-loss and how
many shares to buy.

It is not a fortune teller. It's closer to a very thorough, very consistent
teaching assistant who reads 2,000 report cards every night and tells you
which ten students look most promising *right now*, based on rules anyone
could double-check. It has never met the students, doesn't know if one of them
is lying on their resume, and might be wrong. That's the deal, and the project
says so in giant letters in its own README: **"This is a screen, not advice."**

---

## 2. The whole process, as a story

Imagine you're a talent scout for a big sports league, except instead of
players you're scouting companies, and instead of watching them play you can
only look at numbers on paper. Here's how this program does that job, step by
step.

### Step 1 — Get the paperwork

You can't scout a player you have no stats for. Every trading day, the NSE
itself publishes a file (the exchange's own official record, not some
third-party guess) listing, for every stock: the price it opened at, the price
it closed at, the highest and lowest price of the day, how many shares
changed hands, and — uniquely useful — **what percentage of those shares were
actually "delivered"** (see [Delivery percentage](#delivery-percentage-deliv-pct)
in the glossary; short version: real buying vs. same-day flipping).

The program downloads this file every evening and saves it into one single
file on your computer (a small database, [SQLite](#sqlite)) so it never has to
ask NSE for the same day's data twice. Two years of history for the whole
market costs about 500 downloads total — because one file *is* the whole
market for one day, not one file per company.

### Step 2 — Fix the "fake crash" bug

Here's a trap that ruins amateur stock-screening tools: sometimes a company
rewards its shareholders by **splitting** its stock (turning 1 share worth
₹2,500 into 5 shares worth ₹500 each — same total value, more shares) or
giving a **bonus** (free extra shares). On the exchange's raw data, this looks
*exactly* like the stock crashing 80% overnight, because the price per share
just dropped by 80% — even though nobody actually lost any money.

If you don't fix this, your "which stock crashed" detector gets fooled
constantly, and it silently punishes companies for the good news of rewarding
their shareholders. This program has a dedicated fix
([`adjust.py`](nse-alpha/nsealpha/adjust.py)) that rewrites the *historical*
prices to remove that fake gap — using two independent pieces of evidence
that are cross-checked against each other, so a disagreement gets flagged
instead of silently guessed at. Today's price is never touched — only the
history behind it — so your stop-loss and targets always match the real
number on your broker's screen.

### Step 3 — Throw out the stocks you can't safely trade

Before judging anyone on merit, the program disqualifies stocks that are
technically "on the list" but not realistically tradeable — the equivalent of
a talent scout ignoring a player who's injured or not actually eligible to
play. A stock is dropped if it:

- costs less than ₹20 a share (classic "penny stock" territory — easy to
  manipulate),
- doesn't trade at least ₹5 crore worth of shares a day on average (too thin —
  you might not be able to sell when you want to),
- has less than about a year of price history (there's nothing to judge yet),
- barely traded recently (something's wrong with it),
- has low ["delivery"](#delivery-percentage-deliv-pct) — meaning most of its
  volume is same-day flipping, not real investing,
- is on the exchange's own watchlist for suspicious trading
  ([ASM/GSM](#asmgsm-surveillance)),
- isn't a normal, ordinary company share (it's a gold ETF, a government bond,
  a real-estate trust, etc. — different animals, not compared here).

Every threshold is a number in one settings file
([`config.yaml`](nse-alpha/config.yaml)) that you can loosen or tighten.

### Step 4 — Build the "vital signs" for every stock that's left

For every stock that survives, the program computes about 25 different
statistics from its price history — moving averages, volatility measures,
momentum readings, and so on. Think of this like a doctor taking a patient's
temperature, blood pressure, heart rate, and a dozen other vitals before a
diagnosis. None of these numbers alone tells you much. What matters is how
they're combined next.

### Step 5 — Six independent judges score every stock

This is the heart of the program. Instead of one opinion, six separate
"judges" (the code calls them **factor blocks**) each score every stock from
a totally different angle, blind to what the other judges think. A stock
needs to look good to *most* of them to make the final list — one glowing
report card from a single judge isn't enough. The six judges are:

1. **Technical** — "Does the price chart look like a healthy uptrend right
   now?" (moving averages, RSI, volume, etc.)
2. **Momentum** — "Has this stock been consistently going up, and going up
   *faster than the market average*?"
3. **Fundamentals** — "Is this an actual good business — profitable, not
   drowning in debt, honestly reporting its cash?"
4. **News (sentiment)** — "What have the company's own official filings and
   big institutional trades said about it lately?"
5. **Past trends** — "The last time *this specific stock* looked like this,
   what happened next?"
6. **Institutional flows** — "What is the smart, large money doing in the
   derivatives and ownership data — not what does the chart *look* like?"

Each judge is explained in full, sub-factor by sub-factor, in
[Section 4](#4-the-six-judges-in-full-detail) below.

### Step 6 — Turn six opinions into one number, fairly

You can't just add "RSI of 65" to "ROE of 22%" — they're not the same kind of
number, like adding degrees Fahrenheit to your shoe size. So before combining
anything, the program:

- **Clips the extreme outliers** ([winsorising](#winsorise)) — so one stock
  that went up 500% doesn't single-handedly stretch the entire scale for
  everyone else.
- **Converts every number to "how many standard deviations from the average
  of today's stocks"** ([z-scoring](#z-score)) — this is what makes "a good
  RSI" and "a good ROE" comparable at all. A z-score of +1 always means "one
  notch better than typical," no matter which measurement it started as.
- **Optionally compares each stock only to others in its own sector**
  ([sector-neutral scoring](#sector-neutral)) — so the list doesn't
  accidentally become "the ten best PSU banks" just because banks are having a
  good month.
- **Blends the sub-scores inside each judge, then blends the six judges
  together**, using the weights you set (Technical 22%, Momentum 18%,
  Fundamentals 22%, News 8%, Past trends 18%, Institutional flows 12%, by
  default).
- **Requires a minimum amount of real data** before a stock is allowed to
  rank at all — a stock missing most of its inputs is dropped rather than
  scored on guesswork.

The output is one number per stock, the **composite score**, and the top-N
stocks (10, by default) become the day's picks.

### Step 7 — Check the weather before recommending an umbrella

A model that only ever ranks stocks will *always* hand you a top 10 — even on
the morning of a market crash, because ranking is about who's "least bad,"
not about whether it's safe to be in the market at all. So before showing any
picks, the program checks the market's own health: is the main index (Nifty
50) above its long-term trend line? What fraction of *all* stocks are
currently in an uptrend? Is the market's fear gauge
([India VIX](#india-vix)) elevated? Are big foreign investors buying or
fleeing?

This produces a **RISK-ON / CAUTION / RISK-OFF** verdict, shown loudly at the
top of every report, along with a suggested "how much of your normal position
size should you actually use today" multiplier (100% / 60% / 25%). The picks
still show up even in a risk-off market — but with a clear warning, because
even the best stock in a falling market can still fall.

### Step 8 — Turn a ranking into an actual trade plan

A ranked list is interesting; a trade plan is useful. For each of the top 10,
the program calculates:

- **A stop-loss** — the price at which you admit the trade didn't work and
  get out — sized to that specific stock's own normal daily wiggle (its
  [ATR](#atr-average-true-range)), not an arbitrary "sell if it drops 5%"
  rule that might be way too tight for a wild stock and way too loose for a
  calm one.
- **Two targets** — profit-taking levels, expressed as multiples of how much
  you're risking (see [R-multiple](#r-multiple)).
- **A position size** — literally, how many shares to buy so that if the stop
  gets hit, you lose exactly 1% of your account (by default), not more. This
  is the single habit that keeps a string of losing trades from ever wiping
  you out.
- **A sector cap** — never more than 3 of the top 10 from the same industry,
  so "diversified list of 10 stocks" doesn't secretly mean "one bet on
  cement, ten times."
- **Red flags** — plain-English warnings shown even on the #1 pick: RSI too
  high (overbought), thin trading volume, promoter has pledged their shares
  as loan collateral (a real warning sign), high debt, a recent bad company
  filing, unusually heavy trading volume that might mean insiders know
  something.

### Step 9 — Publish it three different ways

- **A static website** (`site/index.html`) — a permanent, dated snapshot of
  what the model said tonight. Every run also saves an archived copy, so you
  can look back weeks later and check whether the picks actually worked out.
- **An email digest** — the same information, pushed to your inbox, leading
  with *what changed since yesterday* rather than making you re-read all 10
  rows from scratch every evening.
- **An interactive app** (Streamlit) — a workbench where you can drag the six
  judges' weight sliders and watch the *entire market* re-rank in under a
  second, drill into any single stock's chart, and re-run a backtest with
  different settings — all without touching the settings file. (See
  [Section 5](#5-why-the-app-feels-instant) for how that speed trick works.)

### Step 10 — Check your own homework (backtesting)

Before trusting any of this, you should ask: "if I had followed this model's
picks every few days for the last two years, would I have actually made
money?" That's what **backtesting** answers. The program rewinds to a date in
the past, pretends it only knows what was known *that day* (never peeking at
the future), buys the top 10, holds for a few trading days, charges realistic
trading costs, and repeats — walking forward through history one step at a
time. The results get compared against just buying the index and holding.

The project is honest about this test's limits — see
[Section 6](#6-limits-worth-knowing-about) — the two biggest being that
fundamentals in the backtest are today's numbers, not what was actually known
on that historical day, and that the reported profit doesn't currently
subtract capital-gains tax.

---

## 3. Two useful metaphors for the whole system

**Metaphor 1 — The talent-show judging panel.** Six judges, each scoring on a
completely different skill (singing, dancing, stage presence, songwriting,
audience connection, technical difficulty). No single judge's opinion decides
the winner. Scores get normalized so "a 9/10 for singing" and "a 9/10 for
dance" mean the same *relative* thing before they're averaged. A contestant
who's brilliant at one thing but terrible at everything else usually loses to
someone solid across the board — that's exactly how the weighted blend works.

**Metaphor 2 — A hospital triage nurse, not a doctor.** The program doesn't
diagnose *why* a company is good (it can't read a court filing or sense a
liar). It takes vital signs — price, volume, filings, ownership data — and
sorts patients (stocks) by how urgently a doctor (you) should look at them. A
triage score of "high priority" doesn't mean "definitely sick" — it means
"go check this one yourself, with real attention."

---

## 4. The six judges, in full detail

Each judge produces several **sub-factors** — smaller, focused questions —
which get blended into that judge's overall score using the weights in
`config.yaml`. "Higher = better" for every single number described below; the
code deliberately flips anything that would naturally read the opposite way
(e.g., "lower debt is better" gets turned into a "debt health" score where
higher is still better), so nothing downstream has to remember which
direction is good.

### Judge 1 — Technical (default weight 22%)

*The question it asks: "Is this stock's chart showing a real, healthy,
confirmed uptrend right now?"*

- **Trend structure (22% of this judge)** — Picture three moving averages on
  the chart: a fast one (20-day average price), a medium one (50-day), and a
  slow one (200-day). When today's price is above all three, and they're
  neatly stacked fast-on-top, and even the slow line is still climbing —
  that's a real, established uptrend, not a one-day blip. A bonus if a
  ["golden cross"](#golden-crossdeath-cross) just happened; a penalty for a
  "death cross." Also penalizes being *too* stretched above the 50-day
  average — a stock that's run up too far too fast is flagged as risky
  rather than extra good.
- **ADX strength (10%)** — Measures how *strong and decisive* the trend is,
  separately from its direction. A weak, choppy trend scores low even if the
  price happens to be up. A trend that's been running very strong for a very
  long time is treated as possibly "late" and due to slow down.
- **RSI zone (12%)** — [RSI](#rsi-relative-strength-index) is a 0–100 "how
  excited are buyers right now" gauge. The 55–70 range scores best (strong
  but not crazy). Above ~80 is penalized as overheated. Below ~35 is
  penalized as "still falling, don't try to catch it."
- **MACD (13%)** — Checks whether the gap between two moving averages is
  *growing* (accelerating momentum, good) or *shrinking* (losing steam, even
  if price is still nominally up).
- **Bollinger (8%)** — Rewards a stock breaking out of an unusually *quiet*
  period (tight [Bollinger Bands](#bollinger-bands)) rather than one that's
  already been flying for a while — a breakout from calm tends to matter more
  than one from an already-wild stock.
- **Supertrend (10%)** — A single trend-following line that sits below the
  price in an uptrend (safe to hold) or above it in a downtrend (get out).
  Also scores how much safety cushion there is before that line would flip.
- **Volume confirmation (10%)** — Checks that trading *volume* is backing up
  the price move — price rising on real, heavy participation is trustworthy;
  price rising on thin volume is a rumor, not a trend.
- **Breakout proximity (8%)** — Rewards being near, or breaking above, its
  own 52-week or 20-day high. Counter-intuitively, stocks making new highs
  statistically tend to keep making new highs.
- **Delivery quality (7%, India-specific)** — Rewards rising
  [delivery percentage](#delivery-percentage-deliv-pct) happening *at the
  same time* as a rising price — real investors quietly accumulating shares,
  not day-traders churning volume.

### Judge 2 — Momentum (default weight 18%)

*The question it asks: "Has this stock been a consistent winner, and is it
winning faster than the market as a whole?"*

- **1/3/6-month returns and 12-month-minus-last-month momentum** — plain
  price performance over several time windows. The 12-month version
  deliberately *excludes the most recent month*, because academic research on
  stock markets consistently finds the very last month tends to reverse
  short-term — including it just adds noise.
- **Relative strength vs. the market ([Mansfield RS](#relative-strengthmansfield-rs))**
  — in a bull market almost everything goes up, so what actually matters is
  whether a stock is beating the *index*, and whether that gap is still
  widening or already shrinking.
- **Consistency** — rewards a stock that ground out a steady gain over
  months (smoother, more repeatable) over one that jumped 40% in a single
  week on a rumor. Measured by what share of weeks were positive, a
  return-to-volatility ratio, and whether the whole move came from one giant
  single-day gap (penalized) versus being spread out.

### Judge 3 — Fundamentals (default weight 22%)

*The question it asks: "Is this an actual good business, run honestly, and
not egregiously overpriced?"* Split into five sub-parts:

- **Valuation (26%)** — Is the stock cheap or expensive? The headline test is
  [P/E](#pe-price-to-earnings-ratio) *compared to its own sector's typical
  P/E* — a P/E of 40 is expensive for a slow utility but cheap for a rocket-
  ship software company, so comparing to the sector matters more than the
  raw number. Also uses [P/B](#pb-price-to-book-ratio),
  [PEG](#peg-ratio), [EV/EBITDA](#evebitda), and dividend yield. Being *too*
  cheap (P/E under ~4) stops earning extra credit — that's usually a sign
  something's wrong, a classic "value trap."
- **Quality (28%)** — Is this a well-run, financially sound company? Uses
  [ROE](#roe-return-on-equity) and [ROCE](#roce-return-on-capital-employed)
  (the two numbers that separate a genuine compounding business from one that
  destroys shareholder capital), operating and gross margins,
  [interest coverage](#interest-coverage-ratio-icr) (can it comfortably pay
  interest on its debt?), [debt-to-equity](#debt-to-equity-de) (leverage is
  weighted hard — it's how mid-sized Indian companies most commonly blow up),
  and how *consistently* ROE/ROCE have stayed strong over the last several
  years, not just one lucky year.
- **Growth (22%)** — Revenue, profit, and earnings-per-share growth,
  year-over-year and quarter-over-quarter. Growth funded by piling on debt is
  discounted, because that kind of growth tends to be temporary and risky.
- **Cash quality (14%)** — Uses the actual cash-flow statement, not just
  reported profit. [Free cash flow](#free-cash-flow-fcf) (cash the business
  generated after paying for its own equipment and upkeep) and the
  [cash-conversion ratio](#cash-conversion-ratio) — how much of reported
  profit actually shows up as real cash rather than sitting as unpaid
  invoices or unsold inventory. This is the classic Indian small/mid-cap
  failure mode: profit that looks beautiful on paper for years while the
  cash never actually arrives, because it's parked in receivables and
  inventory. Five straight years of cash matching or beating reported profit
  is very hard to fake with accounting tricks.
- **Ownership (10%)** — Promoter (founder/major-shareholder) holding — high
  and stable is a vote of confidence; promoter *pledge* — meaning the
  promoter borrowed money against their own shares as collateral — is a
  warning sign, and a heavy pledge is treated as close to disqualifying;
  institutional (big fund) ownership.

Banks and finance companies are exempted from interest-coverage, gross-margin
and cash-flow-derived metrics, because for them, borrowed money and interest
*are* the product — those ratios would be meaningless noise, not a red flag.

### Judge 4 — News / sentiment (default weight 8%)

*The question it asks: "What have this company's own official disclosures and
big institutional trades actually said lately?"*

Deliberately does **not** scrape news websites or social media — on Indian
small and mid-cap stocks, that channel is dominated by paid stock promotion
and coordinated pump-and-dump groups on messaging apps, and feeding that into
a ranking model is a direct route to recommending exactly the stocks a
manipulator wants you to buy.

- **Announcements (55%)** — Every company must file official disclosures with
  the exchange: an order win, a credit-rating upgrade, a buyback, a
  regulatory approval (positive); an auditor raising concerns, a loan
  default, a promoter's shares getting seized by a lender, a regulator
  notice (negative). Each filing is scored by keyword and *decays* over
  roughly ten trading days — a three-week-old good-news filing is already
  reflected in the price, so it stops mattering as much. One severely bad
  filing is not allowed to be "averaged out" by ten routine good ones.
- **Deals (25%)** — Large ("bulk" and "block") trades that must be publicly
  disclosed. Net buying by named big players is a mild positive signal; heavy
  net selling is a stronger negative one.
- **Results reaction (20%)** — How did the stock actually move in the three
  trading days right after its last quarterly results, compared to the
  overall market? The idea: the market's real-time verdict on a company's
  numbers is more trustworthy than any keyword-based reading of the filing
  text.

### Judge 5 — Past trends (default weight 18%)

*The question it asks, uniquely among the six: "The last time THIS SPECIFIC
stock looked exactly like this, what actually happened next?"* The other five
judges describe the present; this one is the only one that looks at this
particular stock's own personal history.

- **Historical edge (35%)** — For every past day in a stock's history, the
  program computes a compact "setup score" (a simplified version of the
  Technical judge). It then finds every *other* day in that stock's past
  where the setup score was similar to today's, and averages what happened
  in the 10 trading days that followed those similar days. A stock whose
  current pattern has historically led to gains scores well here — a stock
  whose chart looks pretty today but has repeatedly *failed* from this exact
  setup in the past gets marked down, no matter how good it looks right now.
  Care is taken to only ever use information that would genuinely have been
  available at the time (no cheating by peeking at the future).
- **Regime fit (25%)** — Uses the [Hurst exponent](#hurst-exponent), a
  statistical measure of whether a stock's price tends to keep moving in the
  same direction once it starts (trend-following works) or tends to snap
  back toward its average (trend-following is a trap; betting against the
  move works better). Rewards trend-following signals only on stocks that
  actually behave like trend-followers.
- **Seasonality (15%)** — Some Indian stocks genuinely do better or worse in
  certain calendar months (results season, the annual budget announcement,
  monsoon-linked sectors like agriculture). Small effect, small weight.
- **Drawdown health (15%)** — How far the stock currently sits below its
  52-week peak, plus how reliably it has historically climbed back to a new
  high after drawdowns of a similar size.
- **Volatility regime (10%)** — Penalizes a bullish setup that's firing
  *while volatility is expanding* — a sudden pickup in daily price swings
  more often signals distribution (smart money quietly selling into
  strength) than genuine accumulation.

### Judge 6 — Institutional flows (default weight 12%)

*The question it asks: "What is large, sophisticated money actually DOING —
not what does the chart merely look like?"*

- **Open-interest build-up (35%)** — This is the standout idea in this judge.
  [Open interest](#open-interest-oi) is the total number of outstanding
  futures contracts on a stock. If price is rising **and** open interest is
  rising, that's a **"long build-up"** — genuinely new money is entering
  bets on the stock going up, and it scores best. If price is rising **but**
  open interest is *falling*, that's **"short covering"** — people who had
  bet *against* the stock are just closing those losing bets, not anyone
  actually believing in it — and such rallies statistically tend to fizzle.
  On a plain price chart these two situations look absolutely identical;
  only the open-interest data tells them apart, which is exactly why this
  judge exists.
- **Put-call ratio position (15%)** — The
  [put-call ratio](#put-call-ratio-pcr) compares betting *against* the stock
  (puts) to betting *for* it (calls). Read only at the extremes — the middle
  of the range genuinely carries no useful information and is deliberately
  scored as neutral rather than force-fit into a signal.
- **Basis (15%)** — The gap between the futures price and the actual current
  ("spot") price. A small premium is normal and healthy (it costs money to
  hold a leveraged position, so a small premium is just that cost). A
  *discount* on a stock that's otherwise rising is a warning: the rally
  isn't backed by leveraged money, which is the classic setup for a sudden,
  fast unwind.
- **Institutional ownership (35%)** — Quarter-over-quarter change in how much
  of the company big foreign investors (FII), big domestic funds (DII), and
  the promoter own, plus any change in how much of the promoter's stake is
  pledged as loan collateral (rising pledge is heavily penalized).

Only about 200 of NSE's ~2,000 stocks have listed futures/options at all, so
the first three sub-factors above simply don't exist for most companies. For
those, the judge is marked **missing**, not scored as "neutral" — the
program correctly treats "we have no data" as different from "the data says
average," and automatically redistributes that judge's weight onto the other
five so a stock isn't unfairly punished (or rewarded) for something that
can't be measured.

---

## 5. Why the app feels instant

The interactive app splits its work into two halves on purpose. The
expensive half — downloading nothing, but loading prices, fixing corporate
actions, and computing all the vital signs and all six judges' raw
sub-factor scores for the whole market — takes a minute or two and doesn't
depend on the weight sliders at all. It's computed once and cached. The cheap
half — clipping outliers, converting to comparable scores, blending with the
current slider weights, ranking, and sizing positions — takes about 50
milliseconds for the entire market. So moving a slider only ever re-runs the
cheap half, which is why the whole ranked list visibly reshuffles the instant
you drag a slider, instead of making you wait two minutes per nudge.

---

## 6. Limits worth knowing about

Stated plainly, because a tool that hides its own weaknesses is more
dangerous than one that's upfront about them:

- **It has no idea what it doesn't know.** It cannot read a court filing,
  sense that a promoter is lying, or know about a scandal that hasn't hit an
  official filing yet. It only sees price, volume, and disclosed numbers.
- **Fundamentals in a backtest are today's numbers, not what was actually
  known on that historical day** — India doesn't publish a clean historical
  archive of financial ratios, so testing "would this have worked a year
  ago" carries a mild dose of hindsight on the fundamentals judge only. The
  project's own fix for checking this: re-run the backtest with the
  fundamentals judge's weight set to zero and compare.
- **Capital-gains tax is not currently subtracted in the backtest.** Since
  the default hold is only about a week, essentially every gain would be
  taxed at the short-term rate in real life — so the backtest's reported
  returns are somewhat rosier than what you'd actually keep after tax.
- **This system ranks; it does not gate.** A stock with terrible debt or a
  bad governance flag isn't automatically excluded — it's just scored badly
  on that one judge, and enough strength elsewhere can still push it into the
  top 10. A stricter design would treat certain red flags as automatic
  disqualifiers instead of just a bad grade.
- **There is no "sell" side.** The whole system only ever answers "what
  should I consider buying today." It has no concept of a position you
  already hold, and no logic for "you own this, here's why you should sell
  it now."

---

## 7. Glossary — every term in this project, plain English, A–Z

**ADX (Average Directional Index)** — A number from 0–100 that measures how
*strong* a price trend is, regardless of direction. Low ADX = the stock is
drifting sideways with no real trend. High ADX = a strong, decisive move is
underway (in *either* direction — a separate reading, "+DI vs. -DI," says
which way).

**ASM/GSM (surveillance)** — Watchlists NSE itself maintains for stocks
showing unusual, potentially manipulated trading patterns (sudden price
spikes, unusual volume, etc.). A stock on either list gets extra trading
restrictions from the exchange and is excluded from this screen entirely.

**ATR (Average True Range)** — The average size of a stock's daily price
swing (high minus low, roughly) over the last two weeks or so. Used instead
of a flat percentage for stop-losses, because a wildly volatile stock needs
more breathing room than a calm one — using the same fixed percentage for
both would be too tight for one and too loose for the other.

**Backtest** — Rewinding the model to a date in the past and checking what it
would have said *using only information available at that time*, then
comparing those picks against what actually happened, to sanity-check whether
the strategy has any real edge.

**Basis (futures basis)** — The percentage gap between a stock's futures
price and its actual current price. A small premium is normal; a discount on
a rising stock is a warning sign (see Judge 6 above).

**Bhavcopy** — NSE's own official end-of-day data file for every stock — the
single authoritative record this whole project is built on, not a
third-party scrape.

**Block deal / Bulk deal** — A very large single trade (above a size
threshold set by the exchange) that must be publicly disclosed. Useful
because it shows exactly what a specific big player bought or sold, and at
what price.

**Bollinger Bands** — A volatility "envelope" drawn around a stock's price:
one band above, one below, based on how much the price has recently been
swinging. When the bands squeeze tight, the stock has been unusually quiet; a
breakout from that quiet zone is often more significant than a move that
starts from an already-choppy stock.

**Bonus shares** — Free extra shares a company gives existing shareholders
(e.g., "1 bonus share for every 1 you already own" doubles your share count
but not your total value). Distorts raw price history exactly like a stock
split does, and gets corrected the same way (see [Corporate action](#corporate-action)).

**Breadth (market breadth)** — What percentage of *all* stocks in the market
are currently in an uptrend (specifically, trading above their own 50-day
average price). A market can look fine on the headline index while breadth
is actually weak — meaning the gains are concentrated in just a few giant
stocks, which is a fragile kind of rally.

**Cash-conversion ratio** — Cumulative real cash generated by the business
(from the cash-flow statement) divided by cumulative reported profit, over
several years. A ratio near or above 1.0 means reported profit is
trustworthy — it's actually showing up as cash, not just sitting as unpaid
customer invoices.

**Circuit filter / circuit limit** — The maximum percentage a stock is
allowed to move in a single day before the exchange automatically halts
trading in it, to prevent panics and manipulation.

**Composite score** — The single final number for each stock, produced by
blending all six judges' scores together using their configured weights.
This is what determines the final ranking.

**Corporate action** — A catch-all term for company events that change the
share structure without changing the underlying value: stock splits, bonus
shares, and rights issues are the main ones this project handles. See
Step 2 in Section 2.

**Coverage (factor coverage)** — What fraction of the judges/sub-factors
actually have real data for a given stock. A stock with too little coverage
is dropped from the ranking entirely rather than being scored on guesswork.

**Death cross / Golden cross** — See [Golden cross / Death cross](#golden-crossdeath-cross).

**Delivery percentage (DELIV_PCT)** — Of all the shares traded in a stock
today, what percentage were actually transferred into buyers' permanent
holding accounts (demat accounts), rather than being bought and sold again
within the very same day? High and rising delivery % alongside a rising
price is read as real investors accumulating a position — a stronger signal
than price alone, and something NSE happens to publish that not every market
does.

**Demat account** — The electronic account Indian investors hold actual
owned shares in, as opposed to shares that were bought and sold intraday and
never actually "delivered" anywhere.

**Drawdown** — How far a stock (or the whole market) has fallen from its
most recent peak, expressed as a percentage.

**EBIT / EBITDA** — Two measures of a company's profit *before* certain
costs are subtracted, used to compare companies on a more apples-to-apples
basis regardless of how much debt or depreciation they carry. EBIT = profit
before interest and tax. EBITDA = EBIT with depreciation and amortization
(the gradual "wearing out" of equipment and intangible assets, as an
accounting expense) added back too.

**EV/EBITDA** — "Enterprise value" (roughly: what it would cost to buy the
entire company, debt included) divided by EBITDA. A valuation ratio,
similar in spirit to P/E, but less distorted by how much debt a company
carries.

**F&O (Futures & Options)** — The derivatives market — contracts that let
traders bet on a stock's future price without owning the stock itself. Only
about 200 of NSE's ~2,000 listed stocks have these contracts at all.

**FII / DII** — **F**oreign **I**nstitutional **I**nvestors and **D**omestic
**I**nstitutional **I**nvestors — the two big categories of large fund
investors that move Indian markets. Watched both individually and together,
because heavy foreign selling that domestic funds fully absorb is a very
different market condition than heavy foreign selling into no buyers at all.

**Free cash flow (FCF)** — Cash a business actually generated from running
its operations, *minus* what it had to spend on equipment and infrastructure
just to keep operating (capital expenditure). Positive, growing free cash
flow means a company can pay dividends, pay down debt, or reinvest without
needing to borrow more — one of the hardest financial metrics to fake.

**Golden cross / Death cross** — A "golden cross" happens when a stock's
medium-term average price (50-day) crosses *above* its long-term average
(200-day) — a classically bullish signal. A "death cross" is the reverse,
and classically bearish.

**Hurst exponent** — A statistical score (roughly 0 to 1) measuring whether a
stock's price movements tend to keep going the same direction once started
(above ~0.55 — trend-following strategies tend to work on it) or tend to
snap back toward an average (below ~0.45 — betting *against* a big move
tends to work better than following it). Named after the mathematician who
originally used this kind of statistic to study Nile River flood patterns.

**India VIX** — The market's own "fear gauge" — a number derived from options
prices that reflects how much volatility traders expect over the next month.
Low VIX = calm, complacent market. High VIX = traders are bracing for big
swings.

**Interest Coverage Ratio (ICR)** — A company's operating profit (EBIT)
divided by its interest expense. Answers: "how many times over could this
company pay the interest on its debt from its regular profit?" Below about
4x is considered risky — an economic downturn could leave the company
unable to service its own debt.

**JSON** — A simple, universal text format for storing structured data (like
a settings file or a webpage's data payload). Used here to embed the day's
full results inside the static HTML report.

**Lakh / Crore** — Indian number units. 1 lakh = 100,000. 1 crore = 10
million (100 lakh). Financial figures in Indian markets are almost always
quoted this way rather than in millions/billions.

**Look-ahead bias** — The single most common way a backtest lies to you:
accidentally letting the model "see" information from the future when
deciding what to do in the past. This project takes deliberate care to avoid
it — e.g., the Past Trends judge only ever uses forward-return windows that
had *already fully completed* before the day being scored.

**Mansfield RS** — See [Relative strength](#relative-strengthmansfield-rs).

**MACD (Moving Average Convergence Divergence)** — A momentum indicator
built from the *gap* between two moving averages of different speeds. Its
"histogram" — the gap's size — turning from shrinking to growing is read as
early evidence that upward momentum is accelerating.

**Moving average (SMA/EMA)** — A stock's average closing price over the last
N days (e.g., "SMA20" = simple average of the last 20 closes), used to
smooth out day-to-day noise and see the underlying trend. "EMA" (exponential
moving average) is a variant that weights recent days more heavily.

**Open interest (OI)** — The total number of futures/options contracts on a
stock that are currently still open (not yet closed out or expired). Rising
OI means new bets are actively being placed; falling OI means existing bets
are being closed.

**PAT (Profit After Tax)** — A company's bottom-line net profit — revenue
minus every single cost, including taxes. The number most commonly called
simply "profit."

**PCR (Put-Call Ratio)** — The ratio of "put" option activity (bets that a
stock will fall) to "call" option activity (bets it will rise), measured on
open interest. Extremely high or low readings are read as informative;
middling readings are treated as noise.

**P/E (Price-to-Earnings ratio)** — A stock's price divided by its
per-share annual profit. Roughly: "how many years of current profit would it
take to earn back what you paid for the stock, if profit never grew?" Lower
usually means cheaper, but only relative to similar companies — this project
specifically compares each stock's P/E to its own sector's typical P/E rather
than judging the raw number alone.

**P/B (Price-to-Book ratio)** — A stock's price divided by its accounting
"book value" per share (roughly, what the company's assets would be worth if
liquidated and all debts paid off). A rougher, older-fashioned valuation
yardstick than P/E, still widely quoted for banks and asset-heavy companies.

**PEG ratio** — P/E divided by the company's expected growth rate. Adjusts
the "is it cheap" question for how fast the company is actually growing — a
higher P/E can be perfectly reasonable if growth is fast enough.

**Percentile** — Where a stock ranks compared to everyone else, expressed as
"top X%" rather than a raw score — often an easier way to read a ranking than
the underlying z-score number.

**Pledge (promoter pledge)** — When a company's founder/major owner
("promoter") borrows money from a lender and puts up their own company
shares as the collateral securing that loan. Any pledge is treated as a
mild warning; a heavy one is treated as a serious governance red flag,
because if the promoter can't repay, the lender can seize and dump those
shares on the market, crashing the price.

**Position sizing** — Deciding *how many shares* to buy, calculated here so
that if the stop-loss is hit, you lose a fixed, small, pre-decided
percentage of your total account (1%, by default) — not a fixed number of
shares or a fixed rupee amount.

**Promoter** — In Indian company terminology, the founder(s) or the family/
group that holds a controlling stake and effectively runs the company — the
rough equivalent of "insider" ownership in other markets.

**R-multiple** — A way of expressing a profit target relative to the risk
taken on a trade. If you're risking ₹10 per share (entry minus stop-loss),
a "2R" target is ₹20 of profit per share — twice what you risked.

**Regime (market regime)** — The program's overall read on whether the
broad market environment is currently safe, cautious, or dangerous for new
long positions — RISK-ON, CAUTION, or RISK-OFF.

**Relative strength / Mansfield RS** — Not the stock's price alone, but how
the stock is doing *compared to the overall market index*. A stock up 20%
while the index is up 25% is actually underperforming, despite the green
number — relative strength is what catches that.

**Rights issue** — A corporate action where a company offers existing
shareholders the right to buy additional new shares, usually at a discount.
Not currently adjusted for by this project's corporate-action fix (it needs
the issue price, which the parser deliberately skips for now).

**ROCE (Return on Capital Employed)** — Operating profit (EBIT) divided by
the total capital invested in the business (equity plus long-term debt).
Answers: "how efficiently does this company turn the money invested in it
into profit?" — considered one of the best single measures of whether a
business is a genuine compounder.

**ROE (Return on Equity)** — Net profit divided by shareholders' equity
(roughly, the money shareholders have invested in the company). Answers:
"for every rupee shareholders have put in, how much profit is the company
generating?"

**RSI (Relative Strength Index)** — A 0–100 momentum gauge measuring how
strong and fast recent gains have been relative to recent losses. Despite
the similar name, this is *not* the same thing as "relative strength vs. the
market" above — RSI compares a stock only to its own recent self.

**Sector-neutral scoring** — Comparing each stock only against others in its
*own industry sector*, rather than the whole market at once — so a
model doesn't accidentally recommend ten banks just because the whole
banking sector happens to be strong this particular month.

**SEBI** — The Securities and Exchange Board of India, the country's stock
market regulator — the rough equivalent of the SEC in the United States.

**Shareholding pattern** — A quarterly public disclosure every listed Indian
company must file, breaking down exactly what percentage of the company is
owned by promoters, foreign institutions, domestic institutions, and the
general public.

**SQLite** — A type of database that lives in a single ordinary file on your
own computer, rather than requiring a separate server to run. This project
stores its entire local market history in one such file
(`data/market.sqlite`).

**Stop-loss** — A pre-decided price at which you exit a losing trade rather
than hoping it recovers — the single most important piece of trading
discipline this project tries to automate the math for.

**Supertrend** — A trend-following indicator that draws a single line
tracking below the price in an uptrend (a trailing stop-loss level, moving
up as the price does) and above it in a downtrend, flipping sides when the
trend itself flips.

**Turnover (traded turnover)** — The total rupee value of all shares traded
in a stock on a given day (shares traded × price), used as this project's
main measure of how liquid — how easy to buy and sell without moving the
price — a stock actually is.

**UDiFF** — A specific NSE bhavcopy file format (introduced after July 2024)
used as this project's backup data source if the primary daily file is
unavailable for some reason.

**Universe** — The full set of stocks the model is willing to consider
ranking *at all*, after the tradability filters in Step 3 have removed
everything too risky, too illiquid, or too new to judge fairly. Typically a
few hundred to ~1,000 stocks out of NSE's roughly 2,000 listed names.

**Volatility** — How much a stock's price swings around, day to day. Higher
volatility means bigger potential gains *and* bigger potential losses in the
same breath — not inherently good or bad, but something every stop-loss and
position size in this project is explicitly adjusted for.

**Walk-forward (backtest)** — The specific style of backtest this project
runs: step forward through history a few trading days at a time,
re-running the *exact same* model fresh at each step using only data known
up to that point, rather than testing on one single lucky (or unlucky) time
window.

**Winsorise** — Before comparing stocks statistically, clip the most extreme
few percent of values on both ends (e.g., a stock that's up 500% gets capped
at whatever the 98th-percentile stock did) so that one wild outlier can't
single-handedly distort the scale for every other stock being measured
against it.

**Z-score** — A way of re-expressing any number as "how many standard steps
above or below today's average, across all stocks being compared." A
z-score of 0 means exactly average; +1 means one typical step better than
average; -2 means two typical steps worse than average. This is the
mathematical trick that lets the program fairly combine numbers that started
out on completely different scales (a percentage, a ratio, a rupee amount)
into one comparable score.

---

*This document describes the reasoning and mechanics of the code as it
exists in this repository. For the exact numeric thresholds currently in
effect, see [`config.yaml`](nse-alpha/config.yaml) — every number mentioned
above as "roughly" or "about" is a specific, editable value there. For a
running audit of what the project does and doesn't yet cover against a
broader investing-analysis reference, see
[`GAP-ANALYSIS.md`](nse-alpha/GAP-ANALYSIS.md).*
