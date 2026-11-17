"""The workflow profiles (app/workflow/profiles.py) must stay valid CLI for their trainers, and collect_results must
copy the right files out of the run folder. Plain asserts, like the rest of app/test."""
import re
from pathlib import Path

import pytest

from app.workflow import collect_results, workflow
from app.workflow.profiles import PROFILES, get_profile, profile_names

ROOT = Path(__file__).resolve().parents[2]


def _trainer_flags(profile: str) -> set[str]:
    return set(re.findall(r'add_argument\(\s*["\'](--[A-Za-z0-9_-]+)["\']',
                          (ROOT / PROFILES[profile]["script"]).read_text()))


@pytest.mark.parametrize("name", profile_names())
def test_profile_script_exists(name):
    assert (ROOT / PROFILES[name]["script"]).is_file()
    assert PROFILES[name]["checkpoint_dir"].startswith("runs/")


@pytest.mark.parametrize("name", profile_names())
def test_profile_flags_exist_in_the_trainer(name):
    # --headless comes from IsaacLab's AppLauncher, not the trainer's own parser
    missing = set(PROFILES[name]["params"]) - _trainer_flags(name) - {"--headless"}
    assert not missing, f"{name}: unknown flags {sorted(missing)}"


def test_build_cmd_expands_every_value_type():
    cmd = workflow.build_cmd("acro", {"--num_envs": 4096, "--buckets": True, "--runner": False,
                                      "--threat-tail-gap": (1.0, 2.0), "--reward-json": {"los_k": 0.2}},
                             isaac_python="/py")
    assert cmd[:2] == ["/py", "app/training/acro_rtl_train_isaac.py"]
    assert cmd[cmd.index("--num_envs") + 1] == "4096"
    assert "--buckets" in cmd and "--runner" not in cmd            # False drops the flag
    assert cmd[cmd.index("--threat-tail-gap") + 1:cmd.index("--threat-tail-gap") + 3] == ["1.0", "2.0"]
    reward = cmd[cmd.index("--reward-json") + 1]
    assert reward.startswith('{"los_k"')                          # dict -> JSON string


def test_overrides_accept_the_flag_with_or_without_dashes():
    assert workflow.normalize_flag("num_envs") == "--num_envs"
    assert workflow.normalize_flag("--num_envs") == "--num_envs"
    assert workflow.parse_value("4096") == 4096
    assert workflow.parse_value("true") is True
    assert workflow.parse_value("geometry") == "geometry"


def test_each_profile_has_its_own_run_dir():
    assert len({p["checkpoint_dir"] for p in PROFILES.values()}) == len(PROFILES)


def test_collect_results_copies_the_run_files(tmp_path):
    win = tmp_path / "win"
    run = win / "runs" / "angle_evader_workflow"
    run.mkdir(parents=True)
    for name in ("BEST.json", "metrics.csv", "summary.json", "run_config.json"):
        (run / name).write_text("x")
    (run / "model_1.pt").write_text("w")
    (run / "latest_full.pt").write_text("big")

    src, rel = collect_results.resolve_src_dir(win, Path("runs/angle_evader_workflow"))
    assert src == run and rel == Path("runs/angle_evader_workflow")
    copied = collect_results.collect(src, tmp_path / "out" / rel)
    names = {f.name for f in copied}
    assert {"BEST.json", "metrics.csv", "model_1.pt"} <= names
    assert (tmp_path / "out" / rel / "model_1.pt").is_file()


def test_collect_results_missing_run_dir_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_results.resolve_src_dir(tmp_path, Path("runs/nope"))


def test_get_profile_rejects_unknown_names():
    with pytest.raises(KeyError):
        get_profile("quadcopter")
