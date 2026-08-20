"""
report.py — renders the website.

Output is a single self-contained index.html: no CDN, no build step, no server.
Open it from disk, drop it on GitHub Pages, or point a local web server at the
site/ folder. All the data for the page is embedded as one JSON blob and drawn
client-side with vanilla JS + inline SVG.

Visual decisions worth knowing about:
  - Factor breakdown bars are a DIVERGING encoding (blue above average, red
    below, gray at zero) because a block z-score is polarity data, not magnitude.
  - Price sparklines are a single series, so they carry no legend — the card
    title names the stock.
  - Every chart has a hover layer; every number that matters is also present as
    text, so nothing is encoded by colour alone.
  - Light and dark are both selected palettes, not an inverted filter.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BLOCK_LABELS = {
    "technical": "Technical",
    "momentum": "Momentum",
    "fundamental": "Fundamentals",
    "sentiment": "News",
    "trend": "Past trends",
    "flows": "Institutional flows",
}


def _f(x, nd=2, dash="—"):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return dash
        return round(float(x), nd)
    except Exception:
        return dash


def _series_for(frames: Dict[str, pd.DataFrame], sym: str, n: int = 120) -> List[float]:
    f = frames.get(sym)
    if f is None or f.empty:
        return []
    return [round(float(v), 2) for v in f["close"].tail(n).tolist()]


def _data_quality(result) -> dict:
    """
    What the model had to work with. Silent data problems are the main way a
    screen like this misleads you, so they get their own panel rather than a
    log line nobody reads.
    """
    ca = getattr(result, "ca_events", None)
    out = {"ca_total": 0, "ca_recent": 0, "ca_disagree": 0, "ca_examples": []}
    if ca is not None and not ca.empty:
        recent = ca[pd.to_datetime(ca["date"]) >= result.as_of - pd.Timedelta(days=400)]
        out["ca_total"] = int(len(ca))
        out["ca_recent"] = int(len(recent))
        bad = ca[ca["disagrees"]]
        out["ca_disagree"] = int(len(bad))
        ex = (bad if len(bad) else recent).sort_values("date", ascending=False).head(6)
        out["ca_examples"] = [{
            "symbol": r["symbol"],
            "date": pd.Timestamp(r["date"]).strftime("%d %b %Y"),
            "factor": round(float(r["factor"]), 4),
            "source": r["source"],
            "purpose": (str(r["purpose"]) or "")[:70],
            "disagrees": bool(r["disagrees"]),
        } for _, r in ex.iterrows()]

    blocks = [b for b in BLOCK_LABELS]
    cov = {}
    for b in blocks:
        if b in result.ranked.columns:
            cov[b] = round(float(result.ranked[b].notna().mean() * 100))
    out["block_coverage"] = cov
    return out


def build_payload(result, frames: Dict[str, pd.DataFrame],
                  capital: float, backtest: dict | None = None) -> dict:
    picks, ranked, regime, cfg = (result.picks, result.ranked,
                                  result.regime, result.config)

    pick_rows = []
    for sym, r in picks.iterrows():
        pick_rows.append({
            "symbol": sym,
            "rank": int(r.get("final_rank", 0)),
            "sector": r.get("sector", "Unclassified"),
            "composite": _f(r.get("composite"), 3),
            "percentile": _f(r.get("percentile"), 1),
            "blocks": {b: _f(r.get(b), 2) for b in BLOCK_LABELS},
            "coverage": _f(r.get("coverage", 0) * 100, 0),
            "price": _f(r.get("close")),
            "ret_1m": _f(r.get("ret_1m"), 1),
            "ret_3m": _f(r.get("ret_3m"), 1),
            "ret_6m": _f(r.get("ret_6m"), 1),
            "rsi": _f(r.get("rsi14"), 1),
            "adx": _f(r.get("adx"), 1),
            "atr_pct": _f(r.get("atr_pct"), 2),
            "from_52w_high": _f(r.get("pct_from_52w_high"), 1),
            "turnover_cr": _f(r.get("turnover_20d_cr"), 1),
            "delivery": _f(r.get("deliv_ma20"), 1),
            "icr": _f(r.get("_icr"), 1),
            "cash_conversion": _f(r.get("_cash_conversion"), 2),
            "gross_margin": _f(r.get("_gross_margin"), 1),
            "roce": _f(r.get("_roce"), 1),
            "fcf_cum": _f(r.get("_fcf_cumulative"), 0),
            "fcf_pos_years": _f(r.get("_fcf_positive_years"), 0),
            "fcf_years": _f(r.get("_fcf_years"), 0),
            "pe": _f(r.get("_pe"), 1),
            "sector_pe": _f(r.get("_sector_pe"), 1),
            "pb": _f(r.get("_pb"), 2),
            "roe": _f(r.get("_roe"), 1),
            "de": _f(r.get("_de"), 2),
            "rev_growth": _f(r.get("_rev_growth"), 1),
            "eps_growth": _f(r.get("_eps_growth"), 1),
            "div_yield": _f(r.get("_div_yield"), 2),
            "promoter": _f(r.get("_promoter"), 1),
            "pledge": _f(r.get("_pledge"), 1),
            "mcap_cr": _f(r.get("_mcap_cr"), 0),
            "edge_ret": _f(r.get("_edge_ret"), 2),
            "edge_hit": _f((r.get("_edge_hit") or np.nan) * 100, 0),
            "edge_n": int(r.get("_edge_n") or 0),
            "hurst": _f(r.get("_hurst"), 2),
            "pcr": _f(r.get("_pcr_oi"), 2),
            "oi_chg_5d": _f(r.get("_oi_chg_5d"), 1),
            "basis_pct": _f(r.get("_basis_pct"), 2),
            "fii_chg": _f(r.get("_fii_chg"), 2),
            "dii_chg": _f(r.get("_dii_chg"), 2),
            "why": list(r.get("why") or []),
            "flags": list(r.get("flags") or []),
            "plan": {k: _f(r.get(k), 2) for k in
                     ("entry", "entry_zone_low", "entry_zone_high", "stop",
                      "stop_pct", "target1", "target2", "target1_pct",
                      "target2_pct", "risk_per_share")} |
                    {"shares": int(r.get("shares") or 0),
                     "position_value": _f(r.get("position_value"), 0),
                     "capital_at_risk": _f(r.get("capital_at_risk"), 0)},
            "spark": _series_for(frames, sym),
        })

    table_cols = ["composite", "technical", "momentum", "fundamental",
                  "sentiment", "trend", "flows", "close", "ret_1m", "ret_3m",
                  "rsi14", "adx", "turnover_20d_cr", "pct_from_52w_high"]
    tbl = ranked.head(300).copy()
    table_rows = []
    for sym, r in tbl.iterrows():
        table_rows.append({
            "symbol": sym, "rank": int(r.get("rank", 0)),
            "sector": result.sectors.get(sym, "—"),
            **{c: _f(r.get(c), 2) for c in table_cols},
        })

    bench = result.snapshot
    return {
        "as_of": result.as_of.strftime("%d %b %Y"),
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "simulated": bool(getattr(result, "simulated", False)),
        "capital": capital,
        "universe_size": int(result.universe_size),
        "listed_size": int(result.universe_size + len(result.rejected)),
        "ranked_size": int(len(ranked)),
        "regime": {
            "verdict": regime.verdict,
            "exposure": round(regime.exposure * 100),
            "notes": regime.notes,
            "above200": regime.benchmark_above_200,
            "above50": regime.benchmark_above_50,
            "breadth": _f(regime.breadth_pct, 0),
            "vix": _f(regime.vix, 1),
            "bench_ret_1m": _f(regime.bench_ret_1m, 1),
            "fii_20d_cr": _f(regime.fii_20d_cr, 0),
            "dii_20d_cr": _f(regime.dii_20d_cr, 0),
            "fii_long_pct": _f(regime.fii_long_pct, 0),
        },
        "data_quality": _data_quality(result),
        "weights": {b: cfg["weights"][b] for b in BLOCK_LABELS},
        "picks": pick_rows,
        "table": table_rows,
        "rejections": (result.rejected.groupby("reason").size()
                       .sort_values(ascending=False).head(12)
                       .rename("count").reset_index().to_dict("records")
                       if not result.rejected.empty else []),
        "backtest": backtest or {},
        "config_summary": {
            "universe": cfg["universe"]["source"],
            "min_turnover_cr": cfg["filters"]["min_avg_turnover_cr"],
            "min_price": cfg["filters"]["min_price"],
            "sector_neutral": cfg["scoring"]["sector_neutral"],
            "top_n": cfg["scoring"]["top_n"],
            "benchmark": cfg["risk"]["benchmark"],
        },
    }


def render(payload: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=str))
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    # keep a dated copy so you can look back at what the model said
    (out_dir / "archive").mkdir(exist_ok=True)
    stamp = payload["as_of"].replace(" ", "-")
    (out_dir / "archive" / f"{stamp}.html").write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSE Alpha — daily stock ranking</title>
<style>
  .viz-root, :root {
    color-scheme: light;
    --surface-1:#fcfcfb; --plane:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --grid:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,0.10);
    --series-1:#2a78d6; --pos:#2a78d6; --neg:#e34948; --mid:#f0efec;
    --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
    --success-text:#006300;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --plane:#0d0d0d;
      --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
      --series-1:#3987e5; --pos:#3987e5; --neg:#e66767; --mid:#383835;
      --success-text:#0ca30c;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --pos:#3987e5; --neg:#e66767; --mid:#383835;
    --success-text:#0ca30c;
  }

  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--text-primary);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    font-size:15px;line-height:1.55;}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
  header{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;
    justify-content:space-between;margin-bottom:8px}
  h1{font-size:26px;margin:0;letter-spacing:-0.01em}
  h2{font-size:19px;margin:36px 0 12px;letter-spacing:-0.01em}
  h3{font-size:15px;margin:0 0 6px}
  .sub{color:var(--text-secondary);font-size:13px;margin:2px 0 0}
  .card{background:var(--surface-1);border:1px solid var(--border);
    border-radius:12px;padding:16px 18px}
  .tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px;margin:16px 0 4px}
  .tile .label{font-size:12px;color:var(--muted);text-transform:uppercase;
    letter-spacing:.05em}
  .tile .value{font-size:26px;font-weight:600;margin-top:2px}
  .tile .note{font-size:12px;color:var(--text-secondary)}
  .banner{border-radius:12px;padding:14px 18px;margin:16px 0;
    border:1px solid var(--border);background:var(--surface-1);
    display:flex;gap:14px;align-items:flex-start}
  .banner .dot{width:12px;height:12px;border-radius:50%;margin-top:6px;flex:0 0 auto}
  .banner ul{margin:6px 0 0 0;padding-left:18px;color:var(--text-secondary);font-size:13.5px}
  .pill{display:inline-block;font-size:11.5px;padding:2px 9px;border-radius:99px;
    border:1px solid var(--border);color:var(--text-secondary);
    background:var(--plane);white-space:nowrap}
  .picks{display:grid;gap:14px}
  .pick{background:var(--surface-1);border:1px solid var(--border);
    border-radius:12px;padding:16px 18px}
  .pick-head{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap;
    justify-content:space-between}
  .rankno{font-size:13px;font-weight:700;color:var(--surface-1);
    background:var(--text-primary);border-radius:8px;width:28px;height:28px;
    display:flex;align-items:center;justify-content:center;flex:0 0 auto}
  .sym{font-size:19px;font-weight:650;letter-spacing:-0.01em}
  .grid2{display:grid;grid-template-columns:1.15fr 1fr;gap:20px;margin-top:14px}
  @media(max-width:820px){.grid2{grid-template-columns:1fr}}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13px}
  .kv dt{color:var(--muted)}
  .kv dd{margin:0;font-variant-numeric:tabular-nums;text-align:right}
  .why{margin:10px 0 0;padding-left:18px;font-size:13.5px;color:var(--text-secondary)}
  .flags{margin:10px 0 0;padding:10px 12px;border-radius:8px;
    border:1px dashed var(--border);font-size:13px;color:var(--text-secondary)}
  .flags b{color:var(--text-primary)}
  table{width:100%;border-collapse:collapse;font-size:13px;
    font-variant-numeric:tabular-nums}
  th,td{padding:7px 8px;text-align:right;border-bottom:1px solid var(--grid);
    white-space:nowrap}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
  th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
    letter-spacing:.04em;cursor:pointer;user-select:none;position:sticky;top:0;
    background:var(--surface-1)}
  th:hover{color:var(--text-primary)}
  .tblwrap{max-height:520px;overflow:auto;border:1px solid var(--border);
    border-radius:12px;background:var(--surface-1)}
  input[type=search]{background:var(--surface-1);color:var(--text-primary);
    border:1px solid var(--border);border-radius:8px;padding:7px 11px;
    font-size:13.5px;min-width:220px;font-family:inherit}
  .toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
  button.ghost{background:var(--surface-1);color:var(--text-secondary);
    border:1px solid var(--border);border-radius:8px;padding:6px 12px;
    font-size:13px;cursor:pointer;font-family:inherit}
  button.ghost:hover{color:var(--text-primary)}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);
    margin:2px 0 8px;flex-wrap:wrap}
  .legend span{display:flex;gap:6px;align-items:center}
  .swatch{width:11px;height:11px;border-radius:3px;display:inline-block}
  .tip{position:fixed;pointer-events:none;background:var(--surface-1);
    border:1px solid var(--border);border-radius:8px;padding:7px 10px;
    font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.16);opacity:0;
    transition:opacity .1s;z-index:50;max-width:260px}
  .disclaimer{border:1px solid var(--critical);border-radius:12px;padding:14px 18px;
    margin:28px 0 0;font-size:13.5px;color:var(--text-secondary);background:var(--surface-1)}
  .disclaimer b{color:var(--text-primary)}
  details{margin-top:10px}
  summary{cursor:pointer;font-size:13.5px;color:var(--text-secondary)}
  .meth{font-size:13.5px;color:var(--text-secondary)}
  .meth b{color:var(--text-primary)}
  .simbanner{background:var(--warning);color:#0b0b0b;padding:10px 18px;
    border-radius:10px;font-size:13.5px;font-weight:600;margin-bottom:16px}
</style>
</head>
<body>
<div class="wrap">
  <div id="sim"></div>
  <header>
    <div>
      <h1>NSE Alpha</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div style="display:flex;gap:8px">
      <button class="ghost" id="themeBtn">Theme</button>
    </div>
  </header>

  <div id="regime"></div>
  <div class="tiles" id="tiles"></div>

  <h2>Top picks</h2>
  <p class="sub" id="pickIntro"></p>
  <div class="legend">
    <span><i class="swatch" style="background:var(--pos)"></i> above universe average</span>
    <span><i class="swatch" style="background:var(--neg)"></i> below universe average</span>
    <span>Bars show each factor block's z-score vs today's universe; the number beside each bar is the same value.</span>
  </div>
  <div class="picks" id="picks"></div>

  <h2>Full ranking</h2>
  <div class="toolbar">
    <input type="search" id="q" placeholder="Filter by symbol or sector…">
    <span class="sub" id="tblCount"></span>
  </div>
  <div class="tblwrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>

  <h2>How this ranking is built</h2>
  <div class="card meth" id="meth"></div>

  <div class="disclaimer" id="disclaimer"></div>
</div>
<div class="tip" id="tip"></div>

<script>
const DATA = __PAYLOAD__;
const $ = (s,r=document)=>r.querySelector(s);
const fmt = (v,d=2)=> (v===null||v===undefined||v==="—"||Number.isNaN(v)) ? "—"
  : (typeof v==="number" ? v.toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d}) : v);
const inr = v => (v===null||v===undefined||v==="—") ? "—" : "₹"+Number(v).toLocaleString("en-IN",{maximumFractionDigits:2});
const esc = s => String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

/* ---------- theme toggle ---------- */
$("#themeBtn").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : cur === "light" ? "auto" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  $("#themeBtn").textContent = "Theme: " + next;
};

/* ---------- tooltip ---------- */
const tip = $("#tip");
function showTip(e, html){ tip.innerHTML = html; tip.style.opacity = 1;
  const p = 14; let x = e.clientX + p, y = e.clientY + p;
  if (x + 270 > innerWidth) x = e.clientX - 270;
  if (y + 90 > innerHeight) y = e.clientY - 90;
  tip.style.left = x+"px"; tip.style.top = y+"px"; }
function hideTip(){ tip.style.opacity = 0; }

/* ---------- header ---------- */
if (DATA.simulated) $("#sim").innerHTML =
  `<div class="simbanner">SIMULATED DATA — this run used a synthetic market, not live NSE prices.
   Use it to check the plumbing, never to trade.</div>`;
$("#subtitle").textContent =
  `Ranked as of ${DATA.as_of} · ${DATA.listed_size} NSE securities screened → `
  + `${DATA.universe_size} passed the tradability filters → ${DATA.ranked_size} scored`
  + ` · generated ${DATA.generated}`;
$("#themeBtn").textContent = "Theme: auto";

/* ---------- regime banner ---------- */
const R = DATA.regime;
const regimeColor = {"RISK-ON":"var(--good)","CAUTION":"var(--warning)","RISK-OFF":"var(--critical)"}[R.verdict];
$("#regime").innerHTML = `
  <div class="banner">
    <span class="dot" style="background:${regimeColor}"></span>
    <div style="flex:1">
      <h3>Market regime: ${R.verdict} — suggested exposure ${R.exposure}% of your normal size</h3>
      <ul>${R.notes.map(n=>`<li>${esc(n)}</li>`).join("")}</ul>
    </div>
  </div>`;

/* ---------- stat tiles ---------- */
const tiles = [
  ["Benchmark vs 200-DMA", R.above200===null?"—":(R.above200?"Above":"Below"),
   R.above200?"Primary trend intact":"Historically the danger zone"],
  ["Market breadth", R.breadth==="—"?"—":R.breadth+"%", "of stocks above their own 50-DMA"],
  ["India VIX", R.vix, R.vix==="—"?"not available":(R.vix<18?"calm":R.vix<24?"elevated":"fearful")],
  ["Benchmark 1-month", R.bench_ret_1m==="—"?"—":R.bench_ret_1m+"%", "index return over ~22 sessions"],
  ["Universe scored", DATA.ranked_size, `of ${DATA.listed_size} securities screened`],
];
if (R.fii_20d_cr !== "—" && R.fii_20d_cr !== null && R.fii_20d_cr !== undefined)
  tiles.splice(3, 0, ["FII net, 20 sessions",
    "₹" + Number(R.fii_20d_cr).toLocaleString("en-IN") + " cr",
    (R.dii_20d_cr==="—"?"":"DIIs ₹"+Number(R.dii_20d_cr).toLocaleString("en-IN")+" cr")]);
$("#tiles").innerHTML = tiles.map(([l,v,n])=>`
  <div class="card tile"><div class="label">${l}</div>
  <div class="value">${v===null||v===undefined?"—":v}</div>
  <div class="note">${n}</div></div>`).join("");

$("#pickIntro").textContent =
  `Sized for a ₹${Number(DATA.capital).toLocaleString("en-IN")} account risking `
  + `1% per trade, scaled to the regime above. Sector cap applied so the list isn't one bet ten times.`;

/* ---------- sparkline (single series, no legend) ---------- */
function sparkline(vals, w=260, h=54){
  if(!vals || vals.length < 5) return "";
  const min = Math.min(...vals), max = Math.max(...vals), rng = (max-min)||1;
  const pts = vals.map((v,i)=>[ (i/(vals.length-1))*(w-4)+2, h-2-((v-min)/rng)*(h-8) ]);
  const d = pts.map((p,i)=>(i?"L":"M")+p[0].toFixed(1)+" "+p[1].toFixed(1)).join(" ");
  const up = vals[vals.length-1] >= vals[0];
  const col = up ? "var(--pos)" : "var(--neg)";
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img"
    aria-label="Closing price over the last ${vals.length} sessions">
    <path d="${d}" fill="none" stroke="${col}" stroke-width="2"
      stroke-linejoin="round" stroke-linecap="round"/>
    <circle cx="${pts[pts.length-1][0].toFixed(1)}" cy="${pts[pts.length-1][1].toFixed(1)}"
      r="4" fill="${col}" stroke="var(--surface-1)" stroke-width="2"/>
  </svg>
  <div class="sub" style="display:flex;justify-content:space-between;font-size:11.5px">
    <span>${vals.length} sessions</span><span>low ${inr(min)} · high ${inr(max)}</span></div>`;
}

/* ---------- diverging factor bars ---------- */
const LABELS = {technical:"Technical",momentum:"Momentum",fundamental:"Fundamentals",
                sentiment:"News",trend:"Past trends",flows:"Institutional flows"};
function factorBars(blocks, weights){
  const MAX = 3, W = 100;
  return `<div style="display:grid;gap:6px">` + Object.keys(LABELS).map(k=>{
    const v = blocks[k];
    const num = (v==="—"||v===null||v===undefined) ? null : Number(v);
    const pct = num===null ? 0 : Math.max(-1,Math.min(1,num/MAX))*50;
    const col = num===null ? "var(--mid)" : (num>=0 ? "var(--pos)" : "var(--neg)");
    const left = num===null? 50 : (num>=0 ? 50 : 50+pct);
    const wdt  = Math.abs(pct);
    return `<div class="fbar" data-k="${k}" data-v="${num===null?"n/a":num}"
      data-w="${(weights[k]*100).toFixed(0)}"
      style="display:grid;grid-template-columns:98px 1fr 46px;gap:10px;align-items:center;cursor:default">
      <span style="font-size:12.5px;color:var(--text-secondary)">${LABELS[k]}</span>
      <span style="position:relative;height:14px;background:var(--mid);border-radius:4px;display:block">
        <span style="position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--baseline)"></span>
        <span style="position:absolute;left:${left}%;width:${wdt}%;top:0;bottom:0;
          background:${col};border-radius:4px"></span>
      </span>
      <span style="font-size:12.5px;text-align:right;font-variant-numeric:tabular-nums">${num===null?"n/a":num.toFixed(2)}</span>
    </div>`;
  }).join("") + `</div>`;
}

/* ---------- pick cards ---------- */
$("#picks").innerHTML = DATA.picks.map(p=>{
  const P = p.plan;
  const flags = p.flags.length
    ? `<div class="flags"><b>Check before you buy:</b><ul style="margin:6px 0 0;padding-left:18px">
       ${p.flags.map(f=>`<li>${esc(f)}</li>`).join("")}</ul></div>` : "";
  return `<article class="pick">
    <div class="pick-head">
      <div style="display:flex;gap:12px;align-items:center">
        <span class="rankno">${p.rank}</span>
        <div><div class="sym">${esc(p.symbol)}</div>
        <div class="sub">${esc(p.sector)} · ${inr(p.price)} ·
        composite score ${p.composite} — ranked ${p.rank} of ${DATA.ranked_size} scored</div></div>
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap">
        <span class="pill">1M ${p.ret_1m}%</span>
        <span class="pill">3M ${p.ret_3m}%</span>
        <span class="pill">RSI ${p.rsi}</span>
        <span class="pill">ADX ${p.adx}</span>
        <span class="pill">₹${p.turnover_cr} cr/day</span>
      </div>
    </div>

    <div class="grid2">
      <div>
        ${factorBars(p.blocks, DATA.weights)}
        <ul class="why">${p.why.map(w=>`<li>${esc(w)}</li>`).join("")}</ul>
        ${flags}
      </div>
      <div>
        ${sparkline(p.spark)}
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px">
          <dl class="kv">
            <dt>Entry zone</dt><dd>${inr(P.entry_zone_low)}–${inr(P.entry_zone_high)}</dd>
            <dt>Stop loss</dt><dd>${inr(P.stop)} (${P.stop_pct}%)</dd>
            <dt>Target 1</dt><dd>${inr(P.target1)} (+${P.target1_pct}%)</dd>
            <dt>Target 2</dt><dd>${inr(P.target2)} (+${P.target2_pct}%)</dd>
            <dt>Qty</dt><dd>${P.shares}</dd>
            <dt>At risk</dt><dd>${inr(P.capital_at_risk)}</dd>
          </dl>
          <dl class="kv">
            <dt>P/E</dt><dd>${p.pe} <span style="color:var(--muted)">(sec ${p.sector_pe})</span></dd>
            <dt>ROE / ROCE</dt><dd>${p.roe}% / ${p.roce}%</dd>
            <dt>D/E</dt><dd>${p.de}</dd>
            <dt>Interest cover</dt><dd>${p.icr==="—"?"—":p.icr+"×"}</dd>
            <dt>Cash conv.</dt><dd>${p.cash_conversion==="—"?"—":p.cash_conversion+"×"}</dd>
            <dt>FCF+ years</dt><dd>${p.fcf_pos_years==="—"?"—":p.fcf_pos_years+" of "+p.fcf_years}</dd>
          </dl>
        </div>
        <details><summary>Historical behaviour &amp; institutional positioning</summary>
          <dl class="kv" style="margin-top:8px">
            <dt>Avg 10-day return after similar setups</dt><dd>${p.edge_ret}%</dd>
            <dt>Hit rate</dt><dd>${p.edge_hit}%</dd>
            <dt>Sample size</dt><dd>${p.edge_n} days</dd>
            <dt>Hurst exponent</dt><dd>${p.hurst} ${p.hurst!=="—"&&p.hurst>0.55?"(trending)":p.hurst!=="—"&&p.hurst<0.45?"(mean-reverting)":""}</dd>
            <dt>Futures OI change (5d)</dt><dd>${p.oi_chg_5d==="—"?"no derivatives":p.oi_chg_5d+"%"}</dd>
            <dt>Put-call ratio (OI)</dt><dd>${p.pcr}</dd>
            <dt>Futures basis vs spot</dt><dd>${p.basis_pct==="—"?"—":p.basis_pct+"%"}</dd>
            <dt>FII holding change (QoQ)</dt><dd>${p.fii_chg==="—"?"—":p.fii_chg+" pts"}</dd>
            <dt>DII holding change (QoQ)</dt><dd>${p.dii_chg==="—"?"—":p.dii_chg+" pts"}</dd>
            <dt>Factor coverage</dt><dd>${p.coverage}%</dd>
          </dl>
        </details>
      </div>
    </div>
  </article>`;
}).join("");

document.querySelectorAll(".fbar").forEach(el=>{
  el.addEventListener("mousemove", e=>{
    const k = el.dataset.k, v = el.dataset.v, w = el.dataset.w;
    showTip(e, `<b>${LABELS[k]}</b><br>Score ${v} standard deviations vs today's universe`
      + `<br>Weight in composite: ${w}%`);
  });
  el.addEventListener("mouseleave", hideTip);
});

/* ---------- full ranking table ---------- */
const COLS = [
  ["rank","#"],["symbol","Symbol"],["sector","Sector"],["composite","Score"],
  ["technical","Tech"],["momentum","Mom"],["fundamental","Fund"],
  ["sentiment","News"],["trend","Trend"],["close","Price"],
  ["ret_1m","1M %"],["ret_3m","3M %"],["rsi14","RSI"],["adx","ADX"],
  ["turnover_20d_cr","₹cr/day"],["pct_from_52w_high","vs 52wH %"],
];
let sortKey = "rank", sortAsc = true, query = "";
function renderTable(){
  const rows = DATA.table
    .filter(r => !query || (r.symbol+" "+r.sector).toLowerCase().includes(query))
    .sort((a,b)=>{
      const x=a[sortKey], y=b[sortKey];
      const nx = typeof x==="number"?x:(parseFloat(x)); const ny = typeof y==="number"?y:(parseFloat(y));
      const bothNum = !Number.isNaN(nx) && !Number.isNaN(ny);
      const cmp = bothNum ? nx-ny : String(x).localeCompare(String(y));
      return sortAsc ? cmp : -cmp;
    });
  $("#tbl thead").innerHTML = "<tr>" + COLS.map(([k,l])=>
    `<th data-k="${k}">${l}${sortKey===k?(sortAsc?" ▲":" ▼"):""}</th>`).join("") + "</tr>";
  $("#tbl tbody").innerHTML = rows.slice(0,300).map(r=>"<tr>" + COLS.map(([k])=>
    `<td>${r[k]===null||r[k]===undefined?"—":r[k]}</td>`).join("") + "</tr>").join("");
  $("#tblCount").textContent = `${rows.length} shown · click a column to sort`;
  document.querySelectorAll("#tbl thead th").forEach(th=>th.onclick=()=>{
    if(sortKey===th.dataset.k) sortAsc=!sortAsc; else {sortKey=th.dataset.k; sortAsc=(sortKey==="rank");}
    renderTable();
  });
}
$("#q").addEventListener("input", e=>{ query = e.target.value.toLowerCase(); renderTable(); });
renderTable();

/* ---------- methodology ---------- */
const C = DATA.config_summary;
$("#meth").innerHTML = `
<p>Every NSE-listed equity is pulled from the daily bhavcopy, then filtered down to a
tradable set: series EQ only, price above ₹${C.min_price}, at least
₹${C.min_turnover_cr} crore of average daily turnover, a year of price history, and
a delivery percentage high enough to suggest real buyers rather than intraday churn.
Today <b>${DATA.listed_size}</b> securities were screened, <b>${DATA.universe_size}</b>
survived the filters, and <b>${DATA.ranked_size}</b> of those had enough factor coverage
to be scored.</p>

<p>Each survivor is measured on six blocks. Within a block, every sub-factor is
winsorised, converted to a z-score across ${C.sector_neutral?"its sector":"the whole universe"},
and blended; the blocks are then combined at these weights:</p>
<ul>${Object.entries(DATA.weights).map(([k,v])=>
  `<li><b>${LABELS[k]} — ${(v*100).toFixed(0)}%</b>: ${({
    technical:"moving-average structure, ADX, RSI zone, MACD, Bollinger squeeze, Supertrend, OBV and volume confirmation, 52-week-high proximity, and delivery-percentage trend",
    momentum:"1/3/6-month returns, 12-month momentum excluding the last month, Mansfield relative strength against the benchmark, and how consistent the gains have been",
    fundamental:"P/E against the stock's own sector P/E, P/B, PEG, EV/EBITDA and dividend yield for value; ROE, real ROCE (EBIT over capital employed), interest coverage, gross and operating margin and debt-to-equity for quality; revenue and profit growth; free cash flow and cumulative CFO against cumulative PAT for cash quality; promoter holding and pledge",
    sentiment:"NSE corporate filings scored through a keyword lexicon with a 10-day half-life, bulk and block deal direction, and how the stock reacted to its last results",
    trend:"what actually happened in this stock the previous times its setup looked like this, its Hurst exponent, month-of-year seasonality, drawdown recovery record, and whether volatility is expanding",
    flows:"futures open-interest build-up (long build-up scores well, short covering does not), put-call ratio at extremes, futures basis versus spot, and quarter-on-quarter change in FII, DII and promoter holding"
  })[k]}</li>`).join("")}</ul>

<p><b>Why cash flow gets its own sub-block.</b> Reported profit is an opinion; cash
is a fact. The classic accounting failure in Indian small and mid caps is profit
that grows beautifully while operating cash flow does not follow — receivables
balloon, inventory piles up, and the profit turns out to be a ledger entry. So
free cash flow (operating cash flow minus capex) and the cash-conversion ratio
(cumulative CFO over cumulative PAT) are scored separately rather than being
averaged into the other fundamentals, where a strong ROE could hide them.</p>

<p><b>A note on the flows block.</b> Only about 200 NSE stocks have listed
derivatives, so open interest, put-call ratio and basis simply do not exist for
most of the universe. Those stocks are scored as <i>missing</i> on those
sub-factors, not as average — the weights redistribute across the blocks that do
have data. A stock without futures is not "neutrally positioned"; it is
unmeasurable on that axis, and pretending otherwise would quietly bias the
ranking toward large caps.</p>

<p>Ranking is <b>relative</b>: it tells you which stocks look best compared with each
other today, not whether the market is worth being in at all. That second question is
what the regime banner at the top answers, and it is the one that decides whether you
should be deploying capital in the first place.</p>

${(()=>{ const Q = DATA.data_quality || {}; const ex = Q.ca_examples || [];
  const cov = Q.block_coverage || {};
  return `<details><summary>Data quality — corporate actions and factor coverage</summary>
  <p style="margin-top:8px">NSE bhavcopy prices are raw traded prices: a 1:5 split
  shows up as a −80% day unless history is adjusted. This run applied
  <b>${Q.ca_recent||0}</b> corporate-action adjustments
  (${Q.ca_total||0} on file in total). Adjustments are derived from the bhavcopy's own
  PREV_CLOSE column and cross-checked against NSE's corporate-actions filings;
  ${Q.ca_disagree ? `<b>${Q.ca_disagree}</b> disagreed between the two sources and used
  the stated filing ratio — those are worth eyeballing.` : `none disagreed between
  the two sources.`}
  Today's prices are never adjusted, so the entry and stop levels above are the
  real numbers on your terminal.</p>
  ${ex.length?`<table style="margin-top:6px"><thead><tr><th>Symbol</th><th>Ex-date</th>
    <th>Factor</th><th>Source</th><th>Purpose</th></tr></thead><tbody>
    ${ex.map(e=>`<tr><td>${esc(e.symbol)}</td><td>${e.date}</td>
      <td>${e.factor}</td><td>${e.source}${e.disagrees?" ⚠":""}</td>
      <td style="text-align:left;white-space:normal">${esc(e.purpose||"—")}</td></tr>`).join("")}
    </tbody></table>`:""}
  <p style="margin-top:10px">Share of the scored universe with data for each block:
  ${Object.entries(cov).map(([k,v])=>`${LABELS[k]} ${v}%`).join(" · ")}.</p>
  </details>`;})()}

${DATA.rejections.length ? `<details><summary>Why stocks were filtered out (top reasons)</summary>
<ul style="margin-top:8px">${DATA.rejections.map(r=>
  `<li>${esc(r.reason)} — ${r.count} stocks</li>`).join("")}</ul></details>` : ""}

${DATA.backtest && DATA.backtest.summary ? `<details open><summary>Backtest</summary>
<p style="margin-top:8px">${esc(DATA.backtest.summary)}</p></details>` : ""}
`;

$("#disclaimer").innerHTML = `
<b>This is not investment advice.</b> It is a quantitative screen — a way of narrowing
two thousand stocks to ten worth researching properly, nothing more. The model has no
knowledge of pending litigation, related-party transactions, accounting games, promoter
character, or anything else that does not appear in price, volume, filings and ratios.
Every factor here is derived from the past, and the relationship between past and future
returns is unstable by nature. Backtested or historical performance does not predict
future results. Position sizes and stop levels shown are arithmetic illustrations of a
1%-risk rule, not recommendations. Do your own research, consider consulting a
SEBI-registered investment adviser, and never risk money you cannot afford to lose.`;
</script>
</body>
</html>
"""
