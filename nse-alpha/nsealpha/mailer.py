"""
mailer.py — the daily digest that lands in your inbox.

Sends a multipart email: a styled HTML version for normal clients and a plain
text version that stays readable in anything, including a notification preview
on a locked phone. The text part is written properly rather than being an
afterthought, because that is what you will actually see first.

The digest leads with what changed. A ranking that you receive every evening is
only useful if you can see at a glance which names are new, which dropped out,
and whether the market regime shifted — reading ten rows from scratch every day
is how people stop opening the email in week three.

CONFIGURATION
-------------
Credentials come from the environment, or from a `.env` file beside config.yaml.
Never commit that file.

    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=abcd efgh ijkl mnop     # Gmail App Password, not your login
    EMAIL_FROM=you@gmail.com
    EMAIL_TO=you@gmail.com,someone@else.com

For Gmail you need an App Password: Google Account -> Security -> 2-Step
Verification must be ON -> App passwords -> generate one for "Mail". Your normal
Google password will be rejected. Other providers work the same way; Outlook uses
smtp-mail.outlook.com:587, Zoho uses smtp.zoho.in:587.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage
from email.utils import formataddr, formatdate
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nsealpha.mailer")


# ------------------------------------------------------------------- config

@dataclass
class MailConfig:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipients: List[str]
    use_tls: bool = True

    @property
    def ok(self) -> bool:
        return bool(self.host and self.user and self.password and self.recipients)


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader — no extra dependency for five key/value pairs."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


MANAGED_KEYS = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                "SMTP_TLS", "EMAIL_FROM", "EMAIL_TO", "SITE_URL")


def save_env(values: Dict[str, str], path: str | Path = ".env") -> Path:
    """
    Write SMTP settings to `.env`, updating only the keys we manage and leaving
    anything else in the file untouched.

    The file is chmod 0600 because it holds an app password in plaintext. That is
    the same trade every SMTP client makes — there is no way to send mail without
    a credential the machine can read — but it is worth being explicit about
    rather than silently dropping a password into a world-readable file.
    """
    path = Path(path)
    existing: List[str] = []
    if path.exists():
        existing = path.read_text().splitlines()

    seen = set()
    out: List[str] = []
    for line in existing:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in values:
            out.append(f"{key}={values[key]}")
            seen.add(key)
        else:
            out.append(line)

    remaining = [k for k in values if k not in seen]
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# Written by the NSE Alpha app")
        out.extend(f"{k}={values[k]}" for k in remaining)

    path.write_text("\n".join(out).rstrip() + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    # Make the new values visible to this process immediately; load_env only
    # uses setdefault, so it would not otherwise override a stale value.
    for k, v in values.items():
        os.environ[k] = v
    return path


def mail_config(env_file: str | Path = ".env") -> MailConfig:
    load_env(env_file)
    to = os.environ.get("EMAIL_TO", "") or os.environ.get("SMTP_USER", "")
    return MailConfig(
        host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        port=int(os.environ.get("SMTP_PORT", "587")),
        user=os.environ.get("SMTP_USER", ""),
        password=os.environ.get("SMTP_PASSWORD", ""),
        sender=os.environ.get("EMAIL_FROM", "") or os.environ.get("SMTP_USER", ""),
        recipients=[a.strip() for a in to.split(",") if a.strip()],
        use_tls=os.environ.get("SMTP_TLS", "true").lower() != "false",
    )


# ------------------------------------------------------- day-over-day changes

def compare_with_previous(store, result, top_n: int = 10) -> Dict[str, object]:
    """
    What moved since the last run. Uses the `scores` table, which every run
    persists, so this works even if you skipped a few days.
    """
    today = result.as_of.strftime("%Y-%m-%d")
    rows = store.conn.execute(
        "SELECT DISTINCT run_date FROM scores WHERE run_date < ? "
        "ORDER BY run_date DESC LIMIT 1", (today,)).fetchone()
    if not rows:
        return {"first_run": True, "new": [], "dropped": [], "held": [],
                "prev_date": None, "moves": {}}

    prev_date = rows[0]
    prev = store.load_scores(prev_date)
    prev_top = prev.sort_values("rank").head(top_n)["symbol"].tolist()
    prev_rank = dict(zip(prev["symbol"], prev["rank"]))

    now_top = list(result.picks.index)
    new = [s for s in now_top if s not in prev_top]
    dropped = [s for s in prev_top if s not in now_top]
    held = [s for s in now_top if s in prev_top]

    moves = {}
    for i, s in enumerate(now_top, start=1):
        if s in prev_rank:
            moves[s] = int(prev_rank[s]) - i      # positive = moved up
    return {"first_run": False, "new": new, "dropped": dropped, "held": held,
            "prev_date": prev_date, "moves": moves}


# ------------------------------------------------------------------ rendering

def _fmt(v, nd=2, dash="—"):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return dash
        return f"{float(v):,.{nd}f}"
    except Exception:
        return dash


def render_text(result, changes: Dict, site_url: Optional[str] = None) -> str:
    R = result.regime
    L: List[str] = []
    L.append(f"NSE ALPHA — top {len(result.picks)} for {result.as_of:%d %b %Y}")
    L.append("=" * 62)
    L.append("")
    L.append(f"MARKET REGIME: {R.verdict}  (suggested exposure {R.exposure*100:.0f}%)")
    for n in R.notes[:4]:
        L.append(f"  - {n}")
    L.append("")

    if changes.get("first_run"):
        L.append("First run — no previous ranking to compare against.")
    else:
        bits = []
        if changes["new"]:
            bits.append("NEW: " + ", ".join(changes["new"]))
        if changes["dropped"]:
            bits.append("DROPPED OUT: " + ", ".join(changes["dropped"]))
        if not bits:
            bits.append("No change in the top 10 since " + str(changes["prev_date"]))
        L.append("WHAT CHANGED since " + str(changes["prev_date"]))
        for b in bits:
            L.append("  " + b)
    L.append("")
    L.append("-" * 62)

    for sym, r in result.picks.iterrows():
        mv = changes.get("moves", {}).get(sym)
        arrow = ""
        if mv is not None and mv != 0:
            arrow = f"  ({'up' if mv > 0 else 'down'} {abs(mv)})"
        elif sym in changes.get("new", []):
            arrow = "  (NEW)"
        L.append(f"{int(r['final_rank']):2d}. {sym}{arrow}")
        L.append(f"    {r.get('sector', '')}  |  score {r['composite']:+.2f}")
        L.append(f"    Price Rs {_fmt(r.get('close'))}   "
                 f"Entry {_fmt(r.get('entry_zone_low'))}-{_fmt(r.get('entry_zone_high'))}   "
                 f"Stop {_fmt(r.get('stop'))} ({_fmt(r.get('stop_pct'), 1)}%)")
        L.append(f"    Targets {_fmt(r.get('target1'))} / {_fmt(r.get('target2'))}   "
                 f"Qty {int(r.get('shares') or 0)}   "
                 f"Risk Rs {_fmt(r.get('capital_at_risk'), 0)}")
        why = list(r.get("why") or [])[:2]
        for w in why:
            L.append(f"    + {w}")
        for f in list(r.get("flags") or [])[:2]:
            L.append(f"    ! {f}")
        L.append("")

    L.append("-" * 62)
    L.append(f"Scored {len(result.ranked)} of {result.universe_size} tradable stocks "
             f"from {result.universe_size + len(result.rejected)} screened.")
    if site_url:
        L.append(f"Full dashboard: {site_url}")
    L.append("")
    L.append("NOT INVESTMENT ADVICE. This is a quantitative screen — a shortlist to")
    L.append("research, not a recommendation to buy. The model cannot see litigation,")
    L.append("accounting issues or promoter behaviour. Past patterns do not predict")
    L.append("future returns. Never risk money you cannot afford to lose.")
    return "\n".join(L)


def render_html(result, changes: Dict, site_url: Optional[str] = None) -> str:
    R = result.regime
    colour = {"RISK-ON": "#0ca30c", "CAUTION": "#fab219",
              "RISK-OFF": "#d03b3b"}.get(R.verdict, "#898781")

    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    # --- what changed banner ---
    if changes.get("first_run"):
        changed = "<p style='margin:0;color:#52514e'>First run — nothing to compare against yet.</p>"
    else:
        parts = []
        if changes["new"]:
            parts.append("<b>New:</b> " + ", ".join(esc(s) for s in changes["new"]))
        if changes["dropped"]:
            parts.append("<b>Dropped out:</b> " +
                         ", ".join(esc(s) for s in changes["dropped"]))
        if not parts:
            parts.append(f"No change since {esc(changes['prev_date'])}.")
        changed = ("<p style='margin:0;color:#52514e;font-size:14px'>"
                   + "<br>".join(parts) + "</p>")

    cards = []
    for sym, r in result.picks.iterrows():
        mv = changes.get("moves", {}).get(sym)
        if sym in changes.get("new", []):
            badge = ("<span style='background:#0ca30c;color:#fff;font-size:11px;"
                     "padding:2px 7px;border-radius:99px;margin-left:8px'>NEW</span>")
        elif mv:
            up = mv > 0
            badge = (f"<span style='background:{'#e8f5e8' if up else '#fdeaea'};"
                     f"color:{'#006300' if up else '#b02020'};font-size:11px;"
                     f"padding:2px 7px;border-radius:99px;margin-left:8px'>"
                     f"{'▲' if up else '▼'} {abs(mv)}</span>")
        else:
            badge = ""

        why = "".join(
            f"<li style='margin:2px 0'>{esc(w)}</li>" for w in list(r.get("why") or [])[:3])
        flags = list(r.get("flags") or [])[:2]
        flag_html = ""
        if flags:
            flag_html = (
                "<div style='margin-top:8px;padding:8px 10px;background:#fff8e6;"
                "border-left:3px solid #fab219;font-size:13px;color:#52514e'>"
                + "<br>".join("⚠ " + esc(f) for f in flags) + "</div>")

        cards.append(f"""
<tr><td style="padding:0 0 14px 0">
 <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #e1e0d9;
   border-radius:10px;background:#ffffff">
  <tr><td style="padding:14px 16px">
   <div style="font-size:17px;font-weight:600;color:#0b0b0b">
     {int(r['final_rank'])}. {esc(sym)}{badge}</div>
   <div style="font-size:13px;color:#898781;margin-top:2px">
     {esc(r.get('sector',''))} &middot; score {r['composite']:+.2f}
     &middot; ₹{_fmt(r.get('close'))}</div>

   <table width="100%" cellpadding="0" cellspacing="0"
     style="margin-top:10px;font-size:13px;color:#52514e">
    <tr>
      <td style="padding:3px 0">Entry zone</td>
      <td align="right" style="padding:3px 0;color:#0b0b0b">
        ₹{_fmt(r.get('entry_zone_low'))} – ₹{_fmt(r.get('entry_zone_high'))}</td>
      <td width="24"></td>
      <td style="padding:3px 0">Stop</td>
      <td align="right" style="padding:3px 0;color:#b02020">
        ₹{_fmt(r.get('stop'))} ({_fmt(r.get('stop_pct'),1)}%)</td>
    </tr>
    <tr>
      <td style="padding:3px 0">Targets</td>
      <td align="right" style="padding:3px 0;color:#006300">
        ₹{_fmt(r.get('target1'))} / ₹{_fmt(r.get('target2'))}</td>
      <td></td>
      <td style="padding:3px 0">Qty / risk</td>
      <td align="right" style="padding:3px 0;color:#0b0b0b">
        {int(r.get('shares') or 0)} sh &middot; ₹{_fmt(r.get('capital_at_risk'),0)}</td>
    </tr>
   </table>

   <ul style="margin:10px 0 0 0;padding-left:18px;font-size:13px;color:#52514e">{why}</ul>
   {flag_html}
  </td></tr>
 </table>
</td></tr>""")

    site_link = (f"<p style='text-align:center;margin:18px 0'>"
                 f"<a href='{esc(site_url)}' style='background:#2a78d6;color:#fff;"
                 f"padding:10px 22px;border-radius:8px;text-decoration:none;"
                 f"font-size:14px;display:inline-block'>Open the full dashboard</a></p>"
                 if site_url else "")

    notes = "".join(f"<li style='margin:3px 0'>{esc(n)}</li>" for n in R.notes[:4])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f9f9f7;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f9f9f7">
<tr><td align="center" style="padding:22px 12px">
<table width="100%" cellpadding="0" cellspacing="0" style="max-width:620px">

  <tr><td style="padding-bottom:16px">
    <div style="font-size:22px;font-weight:650;color:#0b0b0b">NSE Alpha</div>
    <div style="font-size:13px;color:#898781">
      Top {len(result.picks)} for {result.as_of:%d %b %Y} &middot;
      {len(result.ranked)} of {result.universe_size} tradable stocks scored</div>
  </td></tr>

  <tr><td style="padding-bottom:14px">
    <table width="100%" cellpadding="0" cellspacing="0"
      style="background:#fff;border:1px solid #e1e0d9;border-left:4px solid {colour};
      border-radius:10px">
      <tr><td style="padding:14px 16px">
        <div style="font-size:15px;font-weight:600;color:#0b0b0b">
          Market regime: {R.verdict} — suggested exposure {R.exposure*100:.0f}%</div>
        <ul style="margin:8px 0 0 0;padding-left:18px;font-size:13px;color:#52514e">
          {notes}</ul>
      </td></tr>
    </table>
  </td></tr>

  <tr><td style="padding-bottom:14px">
    <table width="100%" cellpadding="0" cellspacing="0"
      style="background:#fff;border:1px solid #e1e0d9;border-radius:10px">
      <tr><td style="padding:12px 16px">
        <div style="font-size:12px;text-transform:uppercase;letter-spacing:.05em;
          color:#898781;margin-bottom:5px">What changed</div>
        {changed}
      </td></tr>
    </table>
  </td></tr>

  {''.join(cards)}

  <tr><td>{site_link}</td></tr>

  <tr><td style="padding:14px 16px;background:#fff;border:1px solid #d03b3b;
    border-radius:10px;font-size:12px;color:#52514e;line-height:1.6">
    <b style="color:#0b0b0b">This is not investment advice.</b> It is a quantitative
    screen — a way of narrowing two thousand stocks to ten worth researching properly.
    The model cannot see pending litigation, related-party transactions, accounting
    irregularities or promoter behaviour. Every factor is derived from past data, and
    the link between past and future returns is unstable. Position sizes and stops are
    arithmetic illustrations of a 1%-risk rule, not recommendations. Do your own
    research, consider consulting a SEBI-registered investment adviser, and never risk
    money you cannot afford to lose.
  </td></tr>

</table>
</td></tr></table>
</body></html>"""


# --------------------------------------------------------------------- sending

def build_message(cfg: MailConfig, result, changes: Dict,
                  site_url: Optional[str] = None,
                  attach_site: Optional[Path] = None) -> EmailMessage:
    msg = EmailMessage()
    top3 = ", ".join(list(result.picks.index)[:3])
    msg["Subject"] = (f"NSE Alpha {result.as_of:%d %b} · {result.regime.verdict} · "
                      f"{top3}")
    msg["From"] = formataddr(("NSE Alpha", cfg.sender))
    msg["To"] = ", ".join(cfg.recipients)
    msg["Date"] = formatdate(localtime=True)

    msg.set_content(render_text(result, changes, site_url))
    msg.add_alternative(render_html(result, changes, site_url), subtype="html")

    if attach_site and Path(attach_site).exists():
        data = Path(attach_site).read_bytes()
        msg.add_attachment(data, maintype="text", subtype="html",
                           filename=f"nse-alpha-{result.as_of:%Y-%m-%d}.html")
    return msg


def send(cfg: MailConfig, msg: EmailMessage) -> None:
    if not cfg.ok:
        missing = [k for k, v in (("SMTP_HOST", cfg.host), ("SMTP_USER", cfg.user),
                                  ("SMTP_PASSWORD", cfg.password),
                                  ("EMAIL_TO", cfg.recipients)) if not v]
        raise RuntimeError(
            "Email is not configured. Missing: " + ", ".join(missing) +
            ". Copy .env.example to .env and fill it in.")

    ctx = ssl.create_default_context()
    if cfg.port == 465:
        with smtplib.SMTP_SSL(cfg.host, cfg.port, context=ctx, timeout=30) as s:
            s.login(cfg.user, cfg.password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg.host, cfg.port, timeout=30) as s:
            s.ehlo()
            if cfg.use_tls:
                s.starttls(context=ctx)
                s.ehlo()
            s.login(cfg.user, cfg.password)
            s.send_message(msg)
    log.info("digest sent to %s", ", ".join(cfg.recipients))


def send_digest(store, result, site_url: Optional[str] = None,
                attach_site: Optional[Path] = None,
                env_file: str | Path = ".env",
                dry_run: bool = False,
                out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Build and send the digest. With `dry_run=True` nothing is sent and the HTML is
    written to disk instead, which is how you check the layout before wiring cron.
    """
    changes = compare_with_previous(store, result,
                                    top_n=result.config["scoring"]["top_n"])
    cfg = mail_config(env_file)

    if dry_run:
        out_dir = Path(out_dir or "site")
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / "digest_preview.html"
        html_path.write_text(render_html(result, changes, site_url), encoding="utf-8")
        (out_dir / "digest_preview.txt").write_text(
            render_text(result, changes, site_url), encoding="utf-8")
        log.info("dry run — digest written to %s", html_path)
        return html_path

    msg = build_message(cfg, result, changes, site_url, attach_site)
    send(cfg, msg)
    return None
