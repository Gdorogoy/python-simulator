"""Run a training profile on the self-hosted runner (called by .github/workflows/main.yml).

    python3 app/workflow/workflow.py                     # profile: acro (default)
    python3 app/workflow/workflow.py --profile angle
    python3 app/workflow/workflow.py --profile angle --set num_envs=4096 --set total_timesteps=10000000
    python3 app/workflow/workflow.py --profile acro --dry-run     # print the command only
    python3 app/workflow/workflow.py --list

The profile (trainer script, CLI params, run folder) lives in app/workflow/profiles.py. --set overrides one value
without editing anything: booleans as true/false, numbers/JSON as JSON, everything else as a string.
"""
from __future__ import annotations

import argparse
import json
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
    cmd = [isaac_python, prof["script"]]
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="acro", choices=profile_names(), help="which trainer config to run")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="FLAG=VALUE",
                   help="override one profile value, repeatable (the flag may be written with or without --)")
    p.add_argument("--isaac-python", default=ISAAC_PYTHON, help="interpreter that has Isaac Lab (default: $ISAAC_PYTHON)")
    p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    p.add_argument("--print-run-dir", action="store_true", help="print the profile's checkpoint dir and exit (for CI)")
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

    if args.print_run_dir:
        print(get_profile(args.profile)["checkpoint_dir"])
        return 0

    cmd = build_cmd(args.profile, overrides, args.isaac_python)
    print(f"profile: {args.profile}\ncommand: {' '.join(cmd)}\n", flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
