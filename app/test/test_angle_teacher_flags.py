"""The angle trainer's teacher interface.

    --teacher none | policy | los | pid     (explicit: no `auto` fallback, `none` really means no distillation)
    --teacher-checkpoint PATH               the frozen actor for --teacher policy

`--anchor-checkpoint` was the old name of the same flag; it is gone, so add a test that keeps it gone.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "app/training/angle_rtl_train_isaac.py").read_text()


def test_teacher_choices_are_none_policy_los_pid():
    match = re.search(r'"--teacher",\s*choices=\(([^)]*)\)', SRC)
    assert match, "--teacher has no choices"
    assert tuple(re.findall(r'"(\w+)"', match.group(1))) == ("none", "policy", "los", "pid")


def test_teacher_checkpoint_replaced_anchor_checkpoint():
    assert '"--teacher-checkpoint"' in SRC
    assert "anchor_checkpoint" not in SRC
    assert "--anchor-checkpoint" not in SRC


def test_none_teacher_really_disables_distillation():
    # load_anchor must short-circuit on none, and the distill coefficient must go to zero: with anchor=None the
    # rollout would otherwise fall back to the PID's labels and distil from them silently.
    assert 'if mode == "none"' in SRC
    assert 'resolve_teacher(args) != "none"' in SRC
    # ... and it must not force the 8% PID-imitation stage on a random init either
    assert '0.0 if resolve_teacher(args) == "none" else 0.08' in SRC


def test_policy_teacher_reads_the_teacher_checkpoint():
    assert "args.teacher_checkpoint or args.bc_checkpoint_path" in SRC
