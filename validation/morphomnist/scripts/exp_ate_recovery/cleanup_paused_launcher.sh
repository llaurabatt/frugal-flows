#!/usr/bin/env bash
# One-off (2026-10-01): end the paused first grid launcher (pid 376198, screen grid8) once none of the
# fits it started is still running. Ending it earlier could SIGHUP those fits (same screen session).
L=376198
while true; do
  n=0
  for c in $(ps -o pid= --ppid $L 2>/dev/null); do
    ps -o args= --ppid $c | grep -q -E "taskset|micromamba|python" && n=$((n+1))
  done
  [ "$n" -eq 0 ] && break
  echo "$(date -u +%FT%TZ) fits still running under the paused launcher: $n"; sleep 60
done
for c in $(ps -o pid= --ppid $L 2>/dev/null); do kill -9 $c 2>/dev/null; done   # the paused slot-waiting helper
kill -9 $L 2>/dev/null
sleep 2; screen -S grid8 -X quit 2>/dev/null
echo "$(date -u +%FT%TZ) paused launcher $L ended, screen grid8 closed"
