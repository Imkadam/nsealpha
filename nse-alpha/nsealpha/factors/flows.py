"""
flows.py — the sixth block: what large money is doing, as opposed to what the
chart looks like.

Sub-factors (higher = better):

  oi_buildup      The four-quadrant read on futures open interest. Price up with
                  OI up is a LONG BUILD-UP: new money is backing the move, and it
                  scores best. Price up with OI DOWN is SHORT COVERING — the move
                  is people closing losing bets rather than anyone believing in
                  the stock, and such moves usually stall, so it scores poorly
                  despite looking identical on a price chart. Price down with OI
                  up is a short build-up and is penalised hardest.

  pcr_position    Put-call ratio on open interest. Read only at extremes: a high
                  PCR means put writers (who are usually the informed side in
                  Indian options) are comfortable selling downside, which is
                  constructive. A very low PCR means heavy call writing overhead.
                  The middle of the range carries no information and is scored
                  flat, deliberately.

  basis           Futures price versus spot. A premium says leveraged money is
                  paying to be long. A persistent discount on a rising stock says
                  the rally is unsupported by derivatives positioning, which is
                  the classic setup for a fast unwind.

  institutional   Quarter-on-quarter change in FII and DII holding from the
                  shareholding pattern, plus the direction of promoter holding and
                  any change in pledged shares. Slow, but it covers every listed
                  company rather than just the ~200 with derivatives, and rising
                  institutional ownership in a small cap is a genuinely scarce
                  signal.

Coverage note: only F&O-eligible stocks have the first three. Everything else
gets NaN, and scoring.py's coverage logic reweights the remaining blocks rather
than treating a missing derivatives market as a neutral reading. That is the
correct behaviour — a stock without futures is not "averagely positioned", it is
simply unmeasurable on this axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(x, lo, hi):
    return np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)


def compute(ctx) -> pd.DataFrame:
    idx = ctx.snapshot.index
    out = pd.DataFrame(index=idx)

    fno = getattr(ctx, "fno", None)
    holding = getattr(ctx, "shareholding", None)

    # ------------------------------------------------------- OI build-up
    if fno is not None and not fno.empty:
        f = fno.reindex(idx)
        price_chg = ctx.snapshot.get("ret_1m", pd.Series(np.nan, index=idx))
        # a 5-day price move pairs better with a 5-day OI move
        close = ctx.snapshot.get("close")
        prev5 = ctx.snapshot.get("close_prev5")
        px_5d = (close / prev5 - 1) * 100 if prev5 is not None else price_chg

        oi_5d = f["oi_chg_5d_pct"]
        px_sign = np.sign(px_5d.fillna(0))
        oi_sign = np.sign(oi_5d.fillna(0))

        magnitude = np.clip(oi_5d.abs().fillna(0) / 12, 0, 2.0)
        quadrant = pd.Series(np.select(
            [(px_sign > 0) & (oi_sign > 0),      # long build-up
             (px_sign > 0) & (oi_sign <= 0),     # short covering
             (px_sign <= 0) & (oi_sign > 0),     # short build-up
             (px_sign <= 0) & (oi_sign <= 0)],   # long unwinding
            [1.0, -0.35, -1.0, -0.2], default=0.0), index=idx)

        out["oi_buildup"] = (2.5 * quadrant * (0.5 + 0.5 * magnitude)).where(
            oi_5d.notna())

        # ---------------------------------------------------- PCR position
        pcr = f["pcr_oi"]
        pcr_shape = pd.Series(np.select(
            [pcr > 1.6, (pcr > 1.2) & (pcr <= 1.6), (pcr >= 0.7) & (pcr <= 1.2),
             (pcr >= 0.5) & (pcr < 0.7), pcr < 0.5],
            [0.55, 1.00, 0.50, 0.20, 0.05], default=0.5), index=idx)
        out["pcr_position"] = (pcr_shape * 3 +
                               np.clip(f["pcr_trend"].fillna(0) * 2, -1, 1)).where(
            pcr.notna())

        # ---------------------------------------------------------- basis
        spot = ctx.snapshot.get("close")
        basis_pct = (f["fut_close"] / spot.replace(0, np.nan) - 1) * 100
        # a small premium is normal (cost of carry); a discount is the signal
        out["basis"] = np.clip(basis_pct.fillna(np.nan) * 3, -3, 3).where(
            basis_pct.notna() & f["fut_close"].notna())
        out["_basis_pct"] = basis_pct
        out["_pcr_oi"] = pcr
        out["_oi_chg_5d"] = oi_5d
        out["_oi_percentile"] = f["oi_percentile_40d"]
    else:
        for c in ("oi_buildup", "pcr_position", "basis"):
            out[c] = np.nan

    # ------------------------------------------------------ institutional
    if holding is not None and not holding.empty:
        h = holding.reindex(idx)
        fii_chg = pd.to_numeric(h.get("fii_chg"), errors="coerce")
        dii_chg = pd.to_numeric(h.get("dii_chg"), errors="coerce")
        prom_chg = pd.to_numeric(h.get("promoter_chg"), errors="coerce")
        pledge_chg = pd.to_numeric(h.get("pledge_chg"), errors="coerce")
        fii_lvl = pd.to_numeric(h.get("fii"), errors="coerce")

        score = (
            2.0 * np.clip(fii_chg.fillna(0), -3, 3) +
            1.5 * np.clip(dii_chg.fillna(0), -3, 3) +
            1.5 * np.clip(prom_chg.fillna(0), -3, 3) -
            2.5 * np.clip(pledge_chg.fillna(0), -5, 5) +
            0.8 * _clip01(fii_lvl.fillna(0), 1, 25)
        )
        any_data = (fii_chg.notna() | dii_chg.notna() | prom_chg.notna())
        out["institutional"] = score.where(any_data)
        out["_fii_chg"] = fii_chg
        out["_dii_chg"] = dii_chg
        out["_pledge_chg"] = pledge_chg
    else:
        out["institutional"] = np.nan

    return out
