#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# schedule_daily.sh — run the screen every trading evening.
#
# NSE publishes the full bhavcopy (the one with delivery percentages) in the
# early evening IST. 7:00 pm is a safe slot; before ~6:30 pm you will often be
# scoring yesterday's close.
#
# Install (Linux / macOS):
#     chmod +x schedule_daily.sh
#     crontab -e
#     # minute hour dom mon dow   -- 7:00 pm IST, Monday to Friday
#     0 19 * * 1-5  /full/path/to/nse-alpha/schedule_daily.sh
#
# On Windows, use Task Scheduler and point it at:
#     python C:\path\to\nse-alpha\run_daily.py --refresh
#
# NSE is closed on trading holidays; on those days the bhavcopy simply 404s,
# the pipeline reuses the last available session, and nothing breaks.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y-%m-%d)"
LOG="$LOG_DIR/run_$STAMP.log"

# Use a virtualenv if one exists next to the project.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

{
  echo "=== NSE Alpha run started $(date) ==="
  # Drop --email if you only want the website.
  python run_daily.py --refresh --email
  echo "=== finished $(date) ==="
} >> "$LOG" 2>&1

# If the run failed, say so loudly in the log so a silent cron failure doesn't
# look identical to "no email today because the market was closed".
if ! grep -q "Site written to" "$LOG"; then
  echo "!! NSE Alpha run did not produce a site — check the errors above." >> "$LOG"
fi

# Keep a month of logs, bin the rest.
find "$LOG_DIR" -name 'run_*.log' -mtime +31 -delete

# --- optional: publish the site --------------------------------------------
# If site/ is inside a git repo published via GitHub Pages, uncomment this to
# push each evening's ranking automatically.
#
# git add site/ && git commit -m "ranking $STAMP" && git push
