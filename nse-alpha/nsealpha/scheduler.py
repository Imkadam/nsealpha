"""
scheduler.py — turning "email me the top 10 every day" into something that
actually happens when the browser is closed.

WHY THIS ISN'T A STREAMLIT BACKGROUND TASK
------------------------------------------
Streamlit only executes Python while a browser session is open and something
triggers a rerun. A toggle that spawns a timer thread inside the app would look
like it worked, then silently stop the moment you closed the tab, your laptop
slept, or Streamlit recycled the session — which is the worst possible failure
mode for a thing whose entire job is to be reliable at 7pm every evening.

So the button here does not "run the email daily". It **installs an entry in the
operating system's scheduler** — cron on Linux and macOS, Task Scheduler on
Windows — which runs `run_daily.py --refresh --email` at the time you choose,
whether or not the app, the browser, or anything else is open. Removing it takes
the entry out again. The app is a control panel for the OS scheduler, nothing more.

THE ONE THING THIS MODULE MUST NEVER DO
---------------------------------------
Destroy the user's other cron jobs. `crontab` has no "edit one line" operation:
you read the whole file and write the whole file back. So our entry is wrapped in
marker comments, and `merge_crontab()` — a pure function, heavily tested — removes
only the block between those markers and leaves every other line byte-for-byte
intact, including comments, blank lines and environment assignments.

HONEST LIMITS
-------------
  * The job runs **on this machine**. If the computer is asleep or powered off at
    the scheduled time, plain cron does not catch up when it wakes — you simply
    get no email that evening. (A laptop that's usually shut at 7pm is a bad host
    for this; a always-on desktop or a small VPS is a good one.)
  * It runs as your user, with your crontab. It does not need root.
  * On Windows the equivalent is a Task Scheduler entry, created via `schtasks`.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

MARKER_BEGIN = "# >>> nse-alpha daily digest >>>"
MARKER_END = "# <<< nse-alpha daily digest <<<"
WINDOWS_TASK_NAME = "NSE Alpha Daily Digest"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------- platform

def current_platform() -> str:
    s = platform.system().lower()
    if s == "windows":
        return "windows"
    if s in ("linux", "darwin"):
        return "unix"
    return "unsupported"


def scheduler_available() -> Tuple[bool, str]:
    """(available, human explanation)."""
    p = current_platform()
    if p == "windows":
        if shutil.which("schtasks"):
            return True, "Windows Task Scheduler"
        return False, "schtasks was not found on PATH."
    if p == "unix":
        if shutil.which("crontab"):
            return True, "cron"
        return False, ("`crontab` is not installed. On Debian/Ubuntu: "
                       "`sudo apt install cron`. On minimal containers cron is "
                       "usually absent by design — see the manual instructions below.")
    return False, f"Unsupported platform: {platform.system()}"


# -------------------------------------------------------------- command build

def build_command(project_dir: Path | str, python_exe: Optional[str] = None,
                  extra_flags: str = "--refresh --email") -> str:
    """
    The command cron will run. Absolute paths throughout: cron starts with almost
    no environment and a working directory you cannot rely on, so a bare
    `python run_daily.py` is the classic way for a cron job to silently do nothing.
    """
    project_dir = Path(project_dir).resolve()
    python_exe = python_exe or sys.executable
    return (f'cd {_q(str(project_dir))} && '
            f'{_q(str(python_exe))} {_q(str(project_dir / "run_daily.py"))} '
            f'{extra_flags} '
            f'>> {_q(str(project_dir / "logs" / "cron.log"))} 2>&1')


def _q(s: str) -> str:
    """Quote a path for a shell if it needs it."""
    return f'"{s}"' if (" " in s or "'" in s) else s


def cron_line(hour: int, minute: int, weekdays_only: bool, command: str) -> str:
    dow = "1-5" if weekdays_only else "*"
    return f"{minute} {hour} * * {dow} {command}"


def build_block(hour: int, minute: int, weekdays_only: bool,
                command: str) -> str:
    when = "Mon–Fri" if weekdays_only else "every day"
    return "\n".join([
        MARKER_BEGIN,
        f"# Managed by the NSE Alpha app. Edit via the app, or delete this whole",
        f"# block (both marker lines included) to remove the schedule by hand.",
        f"# Runs {when} at {hour:02d}:{minute:02d} local time.",
        cron_line(hour, minute, weekdays_only, command),
        MARKER_END,
    ])


# ------------------------------------------------------- crontab merge (pure)

def merge_crontab(existing: str, block: Optional[str]) -> str:
    """
    Return `existing` with our managed block replaced by `block`
    (or removed entirely when `block` is None).

    Pure and side-effect free so it can be tested exhaustively — this is the
    function that must not eat anyone's other cron jobs. Every line outside our
    markers is preserved exactly, including comments, blanks and `MAILTO=` style
    environment lines.
    """
    lines = existing.splitlines()
    kept: List[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped == MARKER_BEGIN:
            inside = True
            continue
        if stripped == MARKER_END:
            inside = False
            continue
        if not inside:
            kept.append(line)

    # An unterminated begin-marker (hand-edited crontab) would otherwise swallow
    # the rest of the file. Nothing was kept after it, so restore those lines.
    if inside:
        start = next(i for i, l in enumerate(lines) if l.strip() == MARKER_BEGIN)
        kept = lines[:start]

    while kept and not kept[-1].strip():
        kept.pop()

    if block:
        if kept:
            kept.append("")
        kept.extend(block.splitlines())

    out = "\n".join(kept)
    return (out + "\n") if out else ""


def extract_block(existing: str) -> Optional[str]:
    """The managed block currently in the crontab, if any."""
    m = re.search(re.escape(MARKER_BEGIN) + r"(.*?)" + re.escape(MARKER_END),
                  existing, re.S)
    return m.group(0) if m else None


def parse_schedule(block: str) -> Optional[dict]:
    """Read hour / minute / weekday-only back out of an installed block."""
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        minute, hour, _dom, _mon, dow = parts[:5]
        try:
            return {"minute": int(minute), "hour": int(hour),
                    "weekdays_only": dow == "1-5", "command": parts[5]}
        except ValueError:
            return None
    return None


# ------------------------------------------------------------ crontab I/O

def read_crontab() -> str:
    """Current user's crontab. An empty crontab is not an error."""
    try:
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True,
                           timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if r.returncode != 0:
        # "no crontab for <user>" is the normal empty case
        if "no crontab" in (r.stderr or "").lower():
            return ""
        return r.stdout or ""
    return r.stdout or ""


def write_crontab(text: str) -> None:
    r = subprocess.run(["crontab", "-"], input=text, capture_output=True,
                       text=True, timeout=15)
    if r.returncode != 0:
        raise RuntimeError(f"crontab rejected the update: "
                           f"{(r.stderr or r.stdout or '').strip()}")


# ------------------------------------------------------------------- Windows

def _schtasks_install(hour: int, minute: int, weekdays_only: bool,
                      project_dir: Path, python_exe: str,
                      extra_flags: str) -> None:
    project_dir = Path(project_dir).resolve()
    (project_dir / "logs").mkdir(exist_ok=True)
    inner = (f'"{python_exe}" "{project_dir / "run_daily.py"}" {extra_flags}')
    cmd = ["schtasks", "/Create", "/TN", WINDOWS_TASK_NAME, "/TR", inner,
           "/ST", f"{hour:02d}:{minute:02d}", "/F"]
    if weekdays_only:
        cmd += ["/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI"]
    else:
        cmd += ["/SC", "DAILY"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "schtasks failed").strip())


def _schtasks_remove() -> None:
    r = subprocess.run(["schtasks", "/Delete", "/TN", WINDOWS_TASK_NAME, "/F"],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0 and "cannot find" not in (r.stderr or "").lower():
        raise RuntimeError((r.stderr or r.stdout or "schtasks failed").strip())


def _schtasks_query() -> Optional[str]:
    r = subprocess.run(["schtasks", "/Query", "/TN", WINDOWS_TASK_NAME, "/FO", "LIST"],
                       capture_output=True, text=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


# --------------------------------------------------------------- public API

@dataclass
class ScheduleStatus:
    installed: bool
    platform: str
    available: bool
    detail: str
    hour: Optional[int] = None
    minute: Optional[int] = None
    weekdays_only: bool = True
    command: Optional[str] = None
    next_run: Optional[datetime] = None
    last_run: Optional[str] = None
    last_run_ok: Optional[bool] = None
    manual_line: Optional[str] = None


def _next_run(hour: int, minute: int, weekdays_only: bool,
              now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    if weekdays_only:
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
    return candidate


def last_run_info(project_dir: Path | str) -> Tuple[Optional[str], Optional[bool]]:
    """
    When the scheduled job last ran, and whether it looked successful, read from
    the logs the cron wrapper writes. Distinguishing "never ran" from "ran and
    failed" is the whole point — a silent cron failure otherwise looks identical
    to a quiet market day.
    """
    logs = Path(project_dir) / "logs"
    if not logs.exists():
        return None, None
    candidates = sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if not candidates:
        return None, None
    newest = candidates[0]
    when = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%d %b %Y, %H:%M")
    try:
        tail = newest.read_text(errors="replace")[-8000:]
    except OSError:
        return when, None
    ok = "Site written to" in tail
    return when, ok


def status(project_dir: Path | str) -> ScheduleStatus:
    project_dir = Path(project_dir).resolve()
    plat = current_platform()
    available, detail = scheduler_available()
    manual = build_block(19, 0, True, build_command(project_dir))
    last, last_ok = last_run_info(project_dir)

    st = ScheduleStatus(installed=False, platform=plat, available=available,
                        detail=detail, last_run=last, last_run_ok=last_ok,
                        manual_line=manual)

    if not available:
        return st

    if plat == "windows":
        q = _schtasks_query()
        if q:
            st.installed = True
            m = re.search(r"Start Time:\s*([0-9]{1,2}):([0-9]{2})", q)
            if m:
                st.hour, st.minute = int(m.group(1)), int(m.group(2))
                st.next_run = _next_run(st.hour, st.minute, st.weekdays_only)
        return st

    block = extract_block(read_crontab())
    if block:
        st.installed = True
        parsed = parse_schedule(block)
        if parsed:
            st.hour, st.minute = parsed["hour"], parsed["minute"]
            st.weekdays_only = parsed["weekdays_only"]
            st.command = parsed["command"]
            st.next_run = _next_run(st.hour, st.minute, st.weekdays_only)
    return st


def install(project_dir: Path | str, hour: int = 19, minute: int = 0,
            weekdays_only: bool = True, python_exe: Optional[str] = None,
            extra_flags: str = "--refresh --email") -> ScheduleStatus:
    project_dir = Path(project_dir).resolve()
    (project_dir / "logs").mkdir(exist_ok=True)

    available, detail = scheduler_available()
    if not available:
        raise RuntimeError(detail)

    if current_platform() == "windows":
        _schtasks_install(hour, minute, weekdays_only, project_dir,
                          python_exe or sys.executable, extra_flags)
        return status(project_dir)

    command = build_command(project_dir, python_exe, extra_flags)
    block = build_block(hour, minute, weekdays_only, command)
    write_crontab(merge_crontab(read_crontab(), block))
    return status(project_dir)


def remove(project_dir: Path | str) -> ScheduleStatus:
    available, detail = scheduler_available()
    if not available:
        raise RuntimeError(detail)
    if current_platform() == "windows":
        _schtasks_remove()
    else:
        write_crontab(merge_crontab(read_crontab(), None))
    return status(project_dir)
