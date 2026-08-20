"""
sentiment.py — what the tape and the filings are saying.

Deliberately narrow. We do NOT scrape news sites or social media: on Indian
small/mid caps that channel is dominated by paid promotion and Telegram pump
groups, and feeding it into a ranking model is how you end up long the exact
stocks the operators want you long.

Instead we use three sources that are hard to fake:

  announcements   NSE corporate filings, keyword-scored (see ingest.py lexicons).
                  An order win, a rating upgrade, a buyback or a capex
                  announcement is a real, disclosed, dated event. So is an
                  auditor qualification or a pledge invocation. Scores decay
                  exponentially with age (half-life ~10 trading days) because a
                  three-week-old order win is already in the price.
  deals           Bulk and block deals. Net buying by named institutions is a
                  mild positive; heavy net selling is a stronger negative.
  results_reaction How the stock behaved in the three sessions after its last
                  results filing, relative to the market. The market's own
                  verdict on the numbers beats our reading of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 10.0
RESULT_KEYWORDS = ("financial result", "results", "quarterly result",
                   "audited", "unaudited", "board meeting outcome")


def _decay(age_days: pd.Series) -> pd.Series:
    return np.power(0.5, age_days / HALF_LIFE_DAYS)


def compute(ctx) -> pd.DataFrame:
    idx = ctx.snapshot.index
    out = pd.DataFrame(index=idx)
    as_of = ctx.as_of if ctx.as_of is not None else pd.Timestamp.today()

    # -------------------------------------------------------- announcements
    ann = ctx.announcements
    if ann is not None and not ann.empty:
        a = ann.copy()
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        a = a.dropna(subset=["date"])
        a = a[a["date"] <= as_of]
        a["age"] = (as_of - a["date"]).dt.days.clip(lower=0)
        a = a[a["age"] <= 60]
        a["w"] = _decay(a["age"])
        a["weighted"] = a["score"] * a["w"]
        agg = a.groupby("symbol").agg(
            ann_score=("weighted", "sum"),
            ann_count=("weighted", "size"),
            ann_worst=("score", "min"),
            ann_best=("score", "max"))

        # Carry the actual worst filing through, so a red flag can name it and
        # date it instead of saying "something negative happened, go look".
        neg = a[a["score"] <= -2.0]
        if not neg.empty:
            worst = neg.sort_values("score").groupby("symbol").first()
            agg["_ann_worst_subject"] = worst["subject"].reindex(agg.index)
            agg["_ann_worst_age"] = worst["age"].reindex(agg.index)
        else:
            agg["_ann_worst_subject"] = np.nan
            agg["_ann_worst_age"] = np.nan
        # A single severe negative (fraud, default, auditor qualification)
        # should not be averaged away by ten routine positives.
        penalty = np.where(agg["ann_worst"] <= -4.0, -6.0,
                   np.where(agg["ann_worst"] <= -2.5, -2.5, 0.0))
        agg["announcements"] = np.clip(agg["ann_score"], -8, 8) + penalty
        out = out.join(agg[["announcements", "ann_count", "ann_best", "ann_worst",
                            "_ann_worst_subject", "_ann_worst_age"]])
        # No filings at all in 60 days is genuinely neutral, not missing.
        out["announcements"] = out["announcements"].fillna(0.0)
    else:
        out["announcements"] = np.nan

    # ----------------------------------------------------------------- deals
    dl = ctx.deals
    if dl is not None and not dl.empty:
        d = dl.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d = d[(d["date"] <= as_of) & ((as_of - d["date"]).dt.days <= 45)]
        d["value"] = d["qty"].fillna(0) * d["price"].fillna(0)
        d["signed"] = np.where(d["side"].str.upper().str.startswith("B"),
                               d["value"], -d["value"])
        d["w"] = _decay((as_of - d["date"]).dt.days)
        net = (d.assign(sv=d["signed"] * d["w"])
                 .groupby("symbol")["sv"].sum().rename("net_deal_value"))
        gross = d.groupby("symbol")["value"].sum().rename("gross_deal_value")
        joined = pd.concat([net, gross], axis=1).reindex(idx)
        ratio = joined["net_deal_value"] / joined["gross_deal_value"].replace(0, np.nan)
        # scale by how big the deals were versus 20d turnover
        size = joined["gross_deal_value"] / (ctx.snapshot["turnover_20d_cr"] * 1e7 * 20).replace(0, np.nan)
        out["deals"] = (2.5 * ratio.fillna(0) *
                        np.clip(np.log1p(size.fillna(0) * 10), 0, 2)).fillna(0.0)
    else:
        out["deals"] = 0.0

    # ------------------------------------------------------ results reaction
    out["results_reaction"] = _results_reaction(ctx, as_of).reindex(idx).fillna(0.0)
    return out


def _results_reaction(ctx, as_of) -> pd.Series:
    """
    Find each stock's most recent results-related filing in the last 90 days and
    measure its 3-day excess return after that date versus the benchmark.
    The market grades the results; we just read the grade.
    """
    ann = ctx.announcements
    if ann is None or ann.empty:
        return pd.Series(dtype=float)

    a = ann.copy()
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a.dropna(subset=["date"])
    text = (a["subject"].fillna("") + " " + a["detail"].fillna("")).str.lower()
    a = a[text.str.contains("|".join(RESULT_KEYWORDS), regex=True, na=False)]
    a = a[(a["date"] <= as_of) & ((as_of - a["date"]).dt.days <= 90)]
    if a.empty:
        return pd.Series(dtype=float)
    last = a.sort_values("date").groupby("symbol")["date"].last()

    bench = ctx.benchmark.set_index("date")["close"].sort_index()
    res = {}
    for sym, dt in last.items():
        f = ctx.frames.get(sym)
        if f is None or f.empty:
            continue
        c = f["close"]
        after = c.index[c.index >= dt]
        if len(after) < 4:
            continue
        i0 = c.index.get_loc(after[0])
        if i0 == 0 or i0 + 3 >= len(c):
            continue
        stock_ret = c.iloc[i0 + 3] / c.iloc[i0 - 1] - 1
        b = bench.reindex(c.index).ffill()
        if pd.isna(b.iloc[i0 + 3]) or pd.isna(b.iloc[i0 - 1]):
            continue
        bench_ret = b.iloc[i0 + 3] / b.iloc[i0 - 1] - 1
        excess = (stock_ret - bench_ret) * 100
        # decay: a beat from three months ago is stale
        age = (as_of - dt).days
        res[sym] = float(np.clip(excess, -20, 20) / 4) * float(np.power(0.5, age / 45))
    return pd.Series(res, dtype=float)
