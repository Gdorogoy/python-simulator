# Workflow (GitHub Actions)

`.github/workflows/main.yml` runs the training on the self-hosted Windows/WSL runner. The Python side is split in three:

| File | What it does |
|---|---|
| `profiles.py` | One entry per drone style: `script` (trainer), `params` (its CLI flags), `checkpoint_dir` (the run folder). Add a mode by adding an entry. |
| `workflow.py` | Builds and runs the profile's command with the runner's Isaac python. `--profile acro\|angle`, `--set flag=value` for one-off changes, `--dry-run`, `--list`. |
| `collect_results.py` | Copies `BEST/best`, `metrics.csv`, `run_config.json`, `summary.*` and the newest `.pt`/`.onnx` out of the run folder into the CI workspace (uploaded as an artifact). Same `--profile`, or `--run-dir`. |

```bash
python3 app/workflow/workflow.py --list
python3 app/workflow/workflow.py --profile acro --dry-run
python3 app/workflow/workflow.py --profile angle --set num_envs=4096 --set total_timesteps=10000000
python3 app/workflow/collect_results.py --profile angle
```

- `workflow.py` and `collect_results.py` read the run folder from the **same profile**, so they cannot drift.
- Values are typed: `dict` becomes a JSON flag (`--reward-json`), a `list`/tuple spreads over one token per item
  (`--threat-geometry-weights 1.0 0.0`), `True` is a bare flag, `False`/`None` is dropped (set a flag with `--set`).
- Runner paths are env-overridable: `ISAAC_PYTHON` (default `/mnt/e/Isaac/env_isaaclab/Scripts/python.exe`) and
  `WIN_PROJECT` (default the Windows project folder).
- The CI picks the profile from the `workflow_dispatch` input (`profile`), defaulting to `acro` on push; pushes only
  trigger the job when the head commit message contains `[train-drone]`.
- `stable_models/` and `runs/` are **not** synced to the runner: the warm-start checkpoints a profile names must already
  exist in the runner's project folder.
