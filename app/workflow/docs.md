# Workflow (GitHub Actions)

`.github/workflows/main.yml` runs the training on the self-hosted Windows/WSL runner. The Python side is split in three:

| File | What it does |
|---|---|
| `profiles.py` | One entry per drone style: `script` (trainer), `params` (its CLI flags), `checkpoint_dir` (the run folder). Add a mode by adding an entry. |
| `workflow.py` | Builds and runs the profile's command with the runner's Isaac python. `--profile acro\|angle`, `--set flag=value` for one-off changes, `--command "..."` to run a whole pasted command, `--dry-run`, `--print-run-dir`, `--list`. |
| `collect_results.py` | Copies `BEST/best`, `metrics.csv`, `run_config.json`, `summary.*` and the newest `.pt`/`.onnx` out of the run folder into the CI workspace (uploaded as an artifact). Same `--profile`, or `--run-dir`. |

```bash
python3 app/workflow/workflow.py --list
python3 app/workflow/workflow.py --profile acro --dry-run
python3 app/workflow/workflow.py --profile angle --set num_envs=4096 --set total_timesteps=10000000
python3 app/workflow/workflow.py --profile angle --command "-u app/training/angle_rtl_train_isaac.py --headless --num_envs 8192 ..."
python3 app/workflow/collect_results.py --profile angle
```

From the GitHub CLI, the whole command is one `command` input:

```bash
gh workflow run main.yml -f profile=angle \
  -f command="-u app/training/angle_rtl_train_isaac.py --headless --num_envs 8192 --checkpoint-dir runs/my_run ..."
```

- `--command` runs a trainer command pasted as one string instead of the profile's `params`. The interpreter at the
  front is optional: `-u app/training/...` gets `$ISAAC_PYTHON` prepended, a full `/mnt/e/.../python.exe -u ...` is used
  as pasted, and `uv run app/training/...` works too. Backslash-newline continuations and a leading `cd ... &&` are
  tolerated. `--set` cannot be combined with `--command` (edit the pasted command instead).
- In CI the `command` input is passed to `workflow.py --command`. The collect step runs `--print-run-dir`, which reads
  the command's `--checkpoint-dir` (or falls back to the profile's dir), so the artifact comes from the folder the run
  actually wrote.
- `workflow.py` and `collect_results.py` read the run folder from the **same profile** when no command is given, so they
  cannot drift.
- The `angle` profile is only a fallback (its run dir, `--from-rest`, `--headless`); the real flags are the pasted
  command's. Edit a profile in `profiles.py`, or override one value with `--set` when running without `--command`; the
  printed command shows the full settings.
- Values are typed: `dict` becomes a JSON flag (`--reward-json`), a `list`/tuple spreads over one token per item
  (`--threat-geometry-weights 1.0 0.0`), `True` is a bare flag, `False`/`None` is dropped (set a flag with `--set`).
- Runner paths are env-overridable: `ISAAC_PYTHON` (default `/mnt/e/Isaac/env_isaaclab/Scripts/python.exe`) and
  `WIN_PROJECT` (default the Windows project folder).
- The CI picks the profile from the `workflow_dispatch` input (`profile`, defaulting to `acro` on push) and the optional
  `command` input; pushes only trigger the job when the head commit message contains `[train-drone]`.
- `stable_models/` and `runs/` are **not** synced to the runner: the warm-start checkpoints a profile names must already
  exist in the runner's project folder.
