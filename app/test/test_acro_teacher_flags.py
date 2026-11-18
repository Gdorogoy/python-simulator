"""The acro trainer's teacher interface, the same vocabulary as the angle trainer.

    --teacher none | policy | los | pid     (explicit: no `auto`, and no split `checkpoint` / `acro` mode pair)
    --teacher-checkpoint PATH               the frozen actor for --teacher policy

`policy` picks its loader from the checkpoint's own action bounds: this env's acro bounds -> an acro actor, labels used
as-is; anything else -> an angle-style actor, its torques converted with angle_to_acro. That is what the old
`--teacher acro` / `--teacher checkpoint` pair meant, minus the footgun of naming the wrong one.
"""
import re
import types
from pathlib import Path

import pytest
import torch

from app.training import acro_rtl_train_isaac as acro

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "app/training/acro_rtl_train_isaac.py").read_text()
PROFILES = (ROOT / "app/workflow/profiles.py").read_text()


def _choices():
    match = re.search(r'"--teacher",\s*choices=\(([^)]*)\)', SRC)
    assert match, "--teacher has no choices"
    return tuple(re.findall(r'"(\w+)"', match.group(1)))


def test_teacher_choices_are_none_policy_los_pid():
    assert _choices() == ("none", "policy", "los", "pid")


def test_auto_checkpoint_and_acro_modes_are_gone():
    for gone in ("auto", "checkpoint", "acro"):
        assert gone not in _choices()
    assert 'teacher_mode = "checkpoint" if' not in SRC


def test_policy_mode_detects_the_style_from_the_checkpoint_bounds():
    # one mode instead of two: the saved action bounds say whether the file is an acro actor or an angle one
    assert "def teacher_style(" in SRC
    assert "def make_policy_teacher(" in SRC
    assert 'return "acro" if same else "angle"' in SRC


def test_policy_mode_needs_a_checkpoint():
    assert 'if args.teacher == "policy" and not args.teacher_checkpoint' in SRC
    assert "validate_teacher_args(args)" in SRC          # and it fires before Isaac boots


def test_pid_teacher_exists_and_converts_to_acro_units():
    assert "class PidAcroTeacher" in SRC
    assert "act = angle_to_acro(self.u, a, self.low, self.high)" in SRC
    # the PID keeps per-episode integrators, so it must be reset where episodes end
    assert SRC.count("reset_teacher(teacher,") >= 3


def test_acro_profile_uses_the_unified_policy_mode():
    assert '"--teacher": "acro"' not in PROFILES
    assert '"--teacher": "policy"' in PROFILES


# ---- behaviour: the style detection and the early validation are pure enough to exercise without Isaac ----

class _Space:
    low = torch.tensor([-1.0, -6.0, -6.0, -6.0])
    high = torch.tensor([1.0, 6.0, 6.0, 6.0])


def _env():
    return types.SimpleNamespace(unwrapped=types.SimpleNamespace(single_action_space=_Space()))


@pytest.mark.parametrize("bounds, want", [
    ((_Space.low, _Space.high), "acro"),                                            # this env's own rate bounds
    ((torch.tensor([-1.0, -0.5, -1.5, -0.5]), torch.tensor([1.0, 0.5, 1.5, 0.5])), "angle"),
])
def test_teacher_style_reads_the_checkpoint_bounds(tmp_path, bounds, want):
    path = tmp_path / "ckpt.pt"
    torch.save({"action_low": bounds[0], "action_high": bounds[1]}, path)
    assert acro.teacher_style(str(path), _env()) == want


def test_teacher_style_assumes_angle_when_the_file_stores_no_bounds(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"action_low": _Space.low}, path)          # no action_high -> cannot be an acro actor
    assert acro.teacher_style(str(path), _env()) == "angle"


def test_validate_teacher_args_rejects_policy_without_checkpoint():
    with pytest.raises(SystemExit):
        acro.validate_teacher_args(types.SimpleNamespace(teacher="policy", teacher_checkpoint=None))
    acro.validate_teacher_args(types.SimpleNamespace(teacher="policy", teacher_checkpoint="x.pt"))
    acro.validate_teacher_args(types.SimpleNamespace(teacher="los", teacher_checkpoint=None))
    acro.validate_teacher_args(types.SimpleNamespace(teacher="pid", teacher_checkpoint=None))
