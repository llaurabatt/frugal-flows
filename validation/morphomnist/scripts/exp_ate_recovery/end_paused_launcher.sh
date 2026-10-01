#!/usr/bin/env bash
# End a paused (SIGSTOPped) grid launcher once none of the fits it started is still running.
# Usage: bash end_paused_launcher.sh <launcher pid>
# Ends processes by pid only. It never closes a screen by name: `screen -S <name>` matches by prefix,
# and on 2026-10-01 that closed the wrong screen and killed 48 fits. The launcher's screen closes by
# itself once the launcher is gone, and by then no fit is left inside it.
set -u
L=$1
while true; do
  n=0
  for c in $(ps -o pid= --ppid "$L" 2>/dev/null); do
    ps -o args= -p "$c" --ppid "$c" | grep -q -E "micromamba|python" && n=$((n+1))
  done
  [ "$n" -eq 0 ] && break
  echo "$(date -u +%FT%TZ) fits still running under paused launcher $L: $n"; sleep 120
done
for c in $(ps -o pid= --ppid "$L" 2>/dev/null); do kill -9 "$c" 2>/dev/null; done
kill -9 "$L" 2>/dev/null
echo "$(date -u +%FT%TZ) paused launcher $L ended"
