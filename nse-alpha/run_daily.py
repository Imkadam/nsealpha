#!/usr/bin/env python3
"""
run_daily.py — the entry point.

    python run_daily.py --simulate            # synthetic market, no network. Start here.
    python run_daily.py --refresh --backfill  # first real run: pull ~2 years of NSE data
    python run_daily.py --refresh             # every trading day after that
    python run_daily.py --refresh --serve     # ...and open the site in a browser
    python run_daily.py --backtest            # walk-forward validation

Run it after 6:30 pm IST — NSE publishes the full bhavcopy (with delivery data)
in the early evening, and before that you are scoring yesterday.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

import pandas as pd

from nsealpha import panel, report
from nsealpha.pipeline import RunResult, load_config, refresh_data
from nsealpha.pipeline import run as run_pipeline
from nsealpha.store import Store

BANNER = r"""
  _  _ ___ ___    _   _      _
 | \| / __| __|  /_\ | |_ __| |_  __ _
 | .` \__ \ _|  / _ \| | '_ \ ' \/ _` |
 |_|\_|___/___|/_/ \_\_| .__/_||_\__,_|
                       |_|   multi-factor ranking for Indian equities
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank NSE stocks and build the daily site.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--refresh", action="store_true",
                    help="download the latest data from NSE before scoring")
    ap.add_argument("--backfill", action="store_true",
                    help="with --refresh, pull the full history window (slow, once)")
    ap.add_argument("--skip-fundamentals", action="store_true",
                    help="with --refresh, skip the slow per-symbol fundamentals pass")
    ap.add_argument("--simulate", action="store_true",
                    help="generate a synthetic market and score that (no network)")
    ap.add_argument("--backtest", action="store_true",
                    help="run the walk-forward backtest and embed it in the site")
    ap.add_argument("--backtest-periods", type=int, default=30,
                    help="how many rebalance periods to test (each one re-runs the "
                         "whole model, so this is the main cost knob)")
    ap.add_argument("--capital", type=float, default=100000.0,
                    help="account size used for position sizing (default Rs 1,00,000)")
    ap.add_argument("--as-of", default=None, help="score as of this date (YYYY-MM-DD)")
    ap.add_argument("--top", type=int, default=None, help="override how many picks to show")
    ap.add_argument("--serve", action="store_true", help="open the site when done")
    ap.add_argument("--email", action="store_true",
                    help="send the daily digest by email (needs .env — see .env.example)")
    ap.add_argument("--email-preview", action="store_true",
                    help="build the digest and write it to site/digest_preview.html "
                         "without sending anything")
    ap.add_argument("--attach-site", action="store_true",
                    help="with --email, attach the full dashboard HTML")
    ap.add_argument("--site-url", default=None,
                    help="public URL of your published site, linked from the email")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    print(BANNER)
    cfg = load_config(args.config)
    if args.top:
        cfg["scoring"]["top_n"] = args.top

    store = Store(cfg["data"]["db_path"])

    if args.simulate:
        if store.latest_price_date() and store.get_meta("simulated"):
            print("→ using the existing simulated dataset")
        else:
            print("→ generating a synthetic market (no network required)...")
            from nsealpha.simulate import generate
            generate(store)
        print(f"  {store.latest_price_date()} is the latest simulated session")
    elif args.refresh:
        print("→ refreshing data from NSE"
              + (" (full backfill — go make chai, this takes a while)"
                 if args.backfill else ""))
        try:
            refresh_data(cfg, store, full_backfill=args.backfill,
                         skip_fundamentals=args.skip_fundamentals)
        except Exception as e:
            print(f"\n  Data refresh hit a problem: {e}")
            if store.latest_price_date() is None:
                print("  No cached data to fall back on. If NSE is blocking you, try "
                      "setting data.server_mode: true in config.yaml, or run "
                      "--simulate to check the rest of the pipeline works.")
                return 1
            print(f"  Falling back to cached data through {store.latest_price_date()}.")

    if store.latest_price_date() is None:
        print("No data in the database. Run with --refresh --backfill, "
              "or --simulate to try it offline.")
        return 1

    as_of = pd.Timestamp(args.as_of) if args.as_of else None

    print("→ scoring the universe...")
    result: RunResult = run_pipeline(cfg, store, as_of=as_of,
                                     capital=args.capital, verbose=True)
    result.simulated = bool(store.get_meta("simulated")) and args.simulate

    bt = None
    if args.backtest:
        print("→ running the walk-forward backtest (this is the slow one)...")
        from nsealpha.backtest import run_backtest
        try:
            b = run_backtest(cfg, store, max_periods=args.backtest_periods)
            bt = {"summary": b.summary, "stats": b.stats}
            out = Path(cfg["data"]["site_dir"])
            out.mkdir(parents=True, exist_ok=True)
            b.periods.to_csv(out / "backtest_periods.csv", index=False)
            b.trades.to_csv(out / "backtest_trades.csv", index=False)
            print("  " + b.summary)
        except Exception as e:
            print(f"  Backtest failed: {e}")

    print("→ building the site...")
    prices = store.load_prices()
    frames = panel.build_frames(prices, list(result.picks.index), as_of=result.as_of)
    payload = report.build_payload(result, frames, capital=args.capital, backtest=bt)
    payload["simulated"] = result.simulated
    path = report.render(payload, cfg["data"]["site_dir"])

    print("\n" + "=" * 66)
    print(f"  Top {len(result.picks)} for {result.as_of.date()}   "
          f"[market regime: {result.regime.verdict}]")
    print("=" * 66)
    for sym, r in result.picks.iterrows():
        why = (r.get("why") or ["—"])[0]
        print(f"  {int(r['final_rank']):2d}. {sym:<14} score {r['composite']:+.2f}  "
              f"₹{r.get('close', float('nan')):>9,.2f}  "
              f"stop ₹{r.get('stop', float('nan')):>9,.2f}  {why[:44]}")
    print("=" * 66)
    print(f"\n  Site written to: {path.resolve()}")
    print("  Not investment advice — a screen, not a recommendation.\n")

    # ---- email digest -----------------------------------------------------
    if args.email or args.email_preview:
        from nsealpha import mailer
        mailer.load_env(".env")
        site_url = args.site_url or os.environ.get("SITE_URL")
        try:
            preview = mailer.send_digest(
                store, result,
                site_url=site_url,
                attach_site=path if args.attach_site else None,
                dry_run=args.email_preview,
                out_dir=Path(cfg["data"]["site_dir"]))
            if args.email_preview:
                print(f"  Digest preview written to: {preview.resolve()}")
                print("  Check it, then re-run with --email to actually send.")
            else:
                cfgm = mailer.mail_config()
                print(f"  Digest emailed to {', '.join(cfgm.recipients)}")
        except Exception as e:
            print(f"\n  Could not send the digest: {e}")
            if "Username and Password not accepted" in str(e) or "5.7.8" in str(e):
                print("  Gmail rejects normal account passwords over SMTP. You need an "
                      "App Password: Google Account -> Security -> 2-Step Verification "
                      "-> App passwords.")

    if args.serve:
        webbrowser.open(path.resolve().as_uri())
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
