#!/usr/bin/env bash
# Fly the LOS teacher alone through the handoff for a few (cruise_gain, stop_speed) settings; prints 2 lines each.
# Run from the repo root:  bash scripts/teacher_sweep.sh
PY=/mnt/e/Isaac/env_isaaclab/Scripts/python.exe
for cfg in "${CFGS:-4.0 3}"; do
  set -- $cfg
  echo "=== cruise_gain=$1 stop_speed=$2"
  $PY -u app/training/angle_rtl_train_isaac.py --headless \
    --num_envs 8192 --handoff --max-speed 14 \
    --distance-low 15 --distance-high 50 \
    --spawn-dist-low 150 --spawn-dist-high 250 \
    --obs-position-mode height \
    --checkpoint-dir runs/teacher_check_tmp \
    --scratch --teacher los --teacher-check-only \
    --teacher-cruise-gain "$1" --los-stop-speed "$2" \
    --reward rc --reward-json '{"zone_time": 0.0167, "post_hit_steps": 5, "r_zone_entry": 0}' \
    2>&1 | grep -E "TEACHER CHECK|teacher flight|trace env|LOS teacher attitude|Traceback|Error"
done
