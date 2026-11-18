"""Run a training profile on the self-hosted runner (called by .github/workflows/main.yml).

    python3 app/workflow/workflow.py                     # profile: acro (default)
    python3 app/workflow/workflow.py --profile angle
    python3 app/workflow/workflow.py --profile angle --set num_envs=4096 --set total_timesteps=10000000
    python3 app/workflow/workflow.py --profile acro --dry-run     # print the command only
    python3 app/workflow/workflow.py --profile angle --command "-u app/training/angle_rtl_train_isaac.py --headless ..."
    python3 app/workflow/workflow.py --list

The profile (trainer script, CLI params, run folder) lives in app/workflow/profiles.py. --set overrides one value
without editing anything: booleans as true/false, numbers/JSON as JSON, everything else as a string. --command instead
runs a trainer command pasted as one string (the form copied from the runner: "-u app/training/... --flag ..."), so CI
can take the whole command as a workflow input.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.workflow.profiles import ISAAC_PYTHON, get_profile, profile_names  # noqa: E402


def normalize_flag(flag: str) -> str:
    return flag if flag.startswith("--") else f"--{flag}"


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_cmd(profile: str, overrides: dict | None = None, isaac_python: str = ISAAC_PYTHON) -> list[str]:
    """Full command for the profile: python + trainer + every CLI flag (dict -> JSON, list -> one token each)."""
    prof = get_profile(profile)
    params = dict(prof["params"])
    params.update(overrides or {})
    cmd = [isaac_python, "-u", prof["script"]]
    for flag, value in params.items():
        if value is None or value is False:
            continue
        flag = normalize_flag(flag)
        if value is True:
            cmd.append(flag)
        elif isinstance(value, dict):
            cmd.extend([flag, json.dumps(value)])
        elif isinstance(value, (list, tuple)):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def command_argv(text: str, isaac_python: str = ISAAC_PYTHON) -> list[str]:
    """argv for a trainer command pasted as one string.

    Accepts the forms copied from the runner:
        -u app/training/angle_rtl_train_isaac.py --headless ...
        /mnt/e/Isaac/env_isaaclab/Scripts/python.exe -u app/training/angle_rtl_train_isaac.py ...
        app/training/angle_rtl_train_isaac.py --headless ...
    Backslash-newline continuations, plain newlines and a leading `cd ... &&` are tolerated."""
    text = text.replace("\\\r\n", " ").replace("\\\n", " ").replace("\r", " ").replace("\n", " ").strip()
    while text.endswith("\\"):          # a trailing continuation with nothing after it
        text = text[:-1].rstrip()
    if "&&" in text:
        text = text.rsplit("&&", 1)[1].strip()
    parts = shlex.split(text)
    if not parts:
        raise ValueError("--command is empty")
    if parts[:2] == ["uv", "run"]:
        parts = parts[2:]
    name = Path(parts[0]).name.lower()
    if name.startswith("python") or name.endswith(".exe"):
        return parts
    if parts[0] == "-u":
        return [isaac_python, *parts]
    return [isaac_python, "-u", *parts]


def command_run_dir(argv: list[str], profile: str) -> str:
    """The run folder a pasted command writes: its --checkpoint-dir (or --checkpoint-dir=X), else the profile's."""
    for i, token in enumerate(argv):
        if token == "--checkpoint-dir" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--checkpoint-dir="):
            return token.split("=", 1)[1]
    return get_profile(profile)["checkpoint_dir"]


def ensure_headless(argv: list[str]) -> list[str]:
    """Add --headless when the pasted command omits it: the runner is a service with no display, so Isaac Sim would
    try to open a GUI window instead of running."""
    return argv if "--headless" in argv else [*argv, "--headless"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="acro", choices=profile_names(), help="which trainer config to run")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FLAG=VALUE",
                   help="override one profile value, repeatable (the flag may be written with or without --)")
    p.add_argument("--command", default=None, metavar="TEXT",
                   help="run this trainer command instead of the profile's params, pasted as one string (e.g. "
                        "'-u app/training/angle_rtl_train_isaac.py --headless --num_envs 8192 ...'); the python "
                        "interpreter at the front is optional")
    p.add_argument("--isaac-python", default=ISAAC_PYTHON, help="interpreter that has Isaac Lab (default: $ISAAC_PYTHON)")
    p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    p.add_argument("--print-run-dir", action="store_true", help="print the run dir and exit (for CI)")
    p.add_argument("--list", action="store_true", help="list the profiles and exit")
    args = p.parse_args(argv)

    if args.list:
        for name in profile_names():
            prof = get_profile(name)
            print(f"{name:8s} {prof['script']}  ->  {prof['checkpoint_dir']}  ({len(prof['params'])} params)")
        return 0

    overrides: dict[str, object] = {}
    for item in args.overrides:
        if "=" not in item:
            p.error(f"--set expects FLAG=VALUE, got {item!r}")
        flag, _, value = item.partition("=")
        overrides[normalize_flag(flag)] = parse_value(value)

    command = args.command.strip() if args.command else None
    if command and overrides:
        p.error("--set cannot be combined with --command: edit the pasted command instead")
    if command:
        try:
            cmd = command_argv(command, args.isaac_python)
        except ValueError as error:
            p.error(str(error))
        run_dir = command_run_dir(cmd, args.profile)
        had_headless = "--headless" in cmd
        cmd = ensure_headless(cmd)
        source = "pasted command" + ("" if had_headless else " (+ --headless: the runner has no display)")
    else:
        cmd = build_cmd(args.profile, overrides, args.isaac_python)
        run_dir = get_profile(args.profile)["checkpoint_dir"]
        source = f"profile {args.profile}"

    if args.print_run_dir:
        print(run_dir)
        return 0

    print(f"profile: {args.profile}  ({source})\nrun dir: {run_dir}\ncommand: {' '.join(cmd)}\n", flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
