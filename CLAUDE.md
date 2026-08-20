# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where things are

The working directory is `nsealpha/`, but **all project code lives in `nse-alpha/`**.
The uv-managed virtualenv (CPython 3.14, Windows) sits one level up at `.venv/`, so
every command below is run from `nse-alpha/` using `../.venv/Scripts/python.exe`.
`nse-alpha/` is not a git repository.

## Commands

```bash
../.venv/Scripts/python.exe run_daily.py --simulate --serve       # synthetic market, no network - start here
../.venv/Scripts/python.exe run_daily.py --refresh --backfill     # first real run, ~900 days of NSE data (20-60 min)
../.venv/Scripts/python.exe run_daily.py --refresh --serve        # daily run, after 6:30pm IST
../.venv/Scripts/python.exe run_daily.py --backtest               # walk-forward validation
../.venv/Scripts/python.exe run_daily.py --simulate --email-preview  # writes site/digest_preview.{html,txt}, sends nothing
../.venv/Scripts/streamlit.exe run app.py                          # the interactive workbench, localhost:8501
```

Useful flags: `--as-of YYYY-MM-DD` (score history), `--capital`, `--top`,
`--skip-fundamentals` (skips the slow per-symbol pass), `-v`.

### Tests

pytest is **not** installed in the venv. The suite is a plain-assert script with its
own runner:

```bash
../.venv/Scripts/python.exe tests/test_parsers.py
```

Run a single test group by importing it:

```bash
../.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'tests'); import test_parsers as t; t.test_corporate_action_adjustment()"
```

`main()` runs test groups in definition order and stops at the first failing assert.

**Known Windows failure:** `test_cron_line_format` fails on Windows because
`scheduler.build_command()` calls `Path.resolve()`, which rewrites the test's POSIX
fixture path `/opt/nse alpha` as `C:\opt\nse alpha` and drops the quoting the
assertion looks for. It is a fixture/platform mismatch, not a scheduler bug - but it
aborts the run, so the two tests defined after it never execute.

## Architecture

Data flows one way: **NSE archives -> SQLite -> indicators -> six factor blocks ->
z-score & blend -> risk overlay -> three renderers.**

### The two-phase run (this is the load-bearing design decision)

`pipeline.py` splits scoring in two:

- `prepare()` - everything **weight-independent**: load prices, adjust for corporate
  actions, build ~25 indicators for every symbol, compute raw sub-factors for all six
  blocks. Seconds to minutes. Returns a `PreparedRun`.
- `score_from_prepared()` - everything **weight-dependent**: winsorise, z-score,
  blend, rank, apply sector caps, size positions. ~50ms.

`run()` is just the two back to back, so the CLI and backtest are unaffected. The
Streamlit app caches `prepare()` (`@st.cache_data`, 6h TTL) and re-runs only the
cheap half on every slider move. **Anything expensive added to `score_from_prepared()`
kills the sliders.** `app.cfg_fingerprint()` lists exactly which config keys
invalidate the expensive cache - filters and universe, never weights.

### Module map (`nse-alpha/nsealpha/`)

| Module | Role |
|---|---|
| `nseclient.py` | NSE session/cookies, rate limiting, retries, on-disk cache of raw files |
| `ingest.py` | parse bhavcopies (full + UDiFF), indices, filings, corporate actions, deals |
| `adjust.py` | split/bonus adjustment from two cross-checked sources |
| `financials.py` | annual statements (yfinance) into ICR, FCF, cash conversion, real ROCE |
| `flows.py` | F&O OI/PCR/basis, FII-DII, participant OI, quarterly shareholding |
| `store.py` | the single SQLite file; every table and query |
| `universe.py` | hard tradability gates |
| `indicators.py`, `panel.py` | dependency-free indicators; per-symbol frames + cross-sectional snapshot |
| `factors/*.py` | the six blocks |
| `scoring.py`, `risk.py` | standardise/blend/explain; regime, stops, sizing, caps, red flags |
| `pipeline.py`, `backtest.py` | orchestration; walk-forward validation |
| `report.py`, `mailer.py`, `charts.py`, `scheduler.py` | static site, email digest, Plotly figures, cron/schtasks |
| `simulate.py` | synthetic market for offline testing |

### Contracts worth not breaking

- **Factor modules** expose exactly `compute(ctx: FactorContext) -> DataFrame` indexed
  by symbol, with columns that are **raw sub-factor values oriented so higher = better**.
  Shaping (e.g. "RSI 60 beats RSI 85") belongs in the factor module; all
  cross-sectional statistics belong in `scoring.py`. Columns prefixed `_` are
  diagnostics, not scored - `pipeline` carries a few of them through to the report.
- **Missing is not neutral.** A stock with no listed futures scores *missing* on those
  sub-factors and the block weights redistribute (`scoring.min_factor_coverage`).
  Filling missing values with zero or the mean would bias the ranking toward large caps.
- **Corporate-action adjustment runs first**, before anything computes a return, and
  runs *backwards* - today's price is never touched, so printed entry/stop/target
  levels always match a broker terminal. `tests/test_parsers.py::test_corporate_action_adjustment`
  pins the whole behaviour.
- **The backtest calls the same `pipeline.run`** with `as_of` set. Do not fork a
  separate scoring path for it.
- **Every threshold lives in `config.yaml`**; nothing in the scoring path is hardcoded.
  The Streamlit sliders deliberately never write back to it.
- `Store`'s public methods are all wrapped in one re-entrant lock at import time (bottom
  of `store.py`) and the connection uses `check_same_thread=False`. New methods get the
  lock automatically - don't bypass it by touching `self.conn` from outside.

### Three outputs, three jobs

`site/index.html` is a self-contained report (payload JSON injected into one HTML
template, plus a dated copy in `site/archive/`). The email is the push notification and
leads with *what changed* versus the previous run (`mailer.compare_with_previous`). The
Streamlit app is the workbench and is read-only - data refresh stays with `run_daily.py`.

## Diagnosing a run

Empty tables silently turn whole blocks into "missing" rather than erroring. Check what
the local DB actually holds before debugging a score:

```bash
../.venv/Scripts/python.exe -c "import sqlite3;c=sqlite3.connect('data/market.sqlite');print([(t[0],c.execute('select count(*) from '+t[0]).fetchone()[0]) for t in c.execute(\"select name from sqlite_master where type='table'\")])"
```

`announcements`/`deals` empty means the sentiment block is missing; `financial_statements`
empty means interest coverage, FCF and cash conversion are missing; `fno_daily` empty
means three of the four flows sub-factors are missing; `shareholding` empty means the
institutional sub-factor is missing. A partially-finished `--refresh` is the usual cause;
the per-symbol fundamentals and statements passes are the slowest and easiest to interrupt.

`cache/` holds raw NSE downloads and is safe to delete (it re-downloads).
`data/market.sqlite` is hundreds of MB and expensive to rebuild - don't delete it casually.

## Context

- `README.md` explains the reasoning behind every factor and threshold; read the
  relevant section before changing a scoring curve.
- `GAP-ANALYSIS.md` is a live audit against the reference manual the model is meant to
  implement. The open items are deliberate, known gaps - notably no capital-gains tax in
  the backtest (which flatters short holds), no checklist *gate* (the system ranks where
  the guide gates), and no sell side / portfolio concept at all.
- This is a screen, not advice. Keep the disclaimer language intact in the site, email
  and README.
