#!/bin/bash
# run_all.py was started before the tractability guard existed. Let it finish
# E1-E4 (all cheap for the exact solver), stop it before E5 would stall, then
# resume the remainder with solver="auto".
LOG=/home/claude/pactrepo/results/rerun_all.log
cd /home/claude/pactrepo/experiments
for i in $(seq 1 1400); do
  if grep -q "E5 horizon law" "$LOG" 2>/dev/null; then
    echo "E4 complete; stopping legacy run_all before E5"
    pkill -f "run_all.py" 2>/dev/null
    sleep 3
    break
  fi
  if ! pgrep -f "run_all.py" >/dev/null 2>&1; then
    echo "run_all.py exited on its own"
    break
  fi
  sleep 5
done
echo "=== resuming with guarded solver selection ==="
PYTHONPATH=/home/claude/pactrepo python3 -u run_rest.py
echo "HANDOFF_EXIT=$?"
