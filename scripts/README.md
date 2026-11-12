# scripts

Parity tests between the numpy reference implementation and the batched torch / Isaac versions. Run them after
changing anything in `app/dynamics` or `app/control/pid.py` / `torch_pid.py`.

| Script | Needs Isaac? | What it checks |
|---|---|---|
| `isaac_lab/diff_test_physics.py` | no | `dynamics.angle_style_torch_methods` vs `dynamics.angle_style_methods`: mixer inversion, motor lag, thrust, torque, drag and wind on 64 random cases, float64, tolerance 1e-9. |
| `isaac_lab/diff_test_pid.py` | no | `TorchPIDController` vs `PIDController`, batched vs looped, 32 envs x 20 steps, tolerance 1e-5. |
| `isaac_lab/diff_test_acro.py` | no | `TorchRateController` vs `RateController` (with and without gyro feedforward, float64, 1e-9), plus a closed-loop 360 deg roll flip on the numpy acro plant. |
| `isaac_lab/diff_test_trajectory.py` | yes | The same action sequence through the numpy integrator and the PhysX-integrated Isaac env (one env). Reports per-step divergence in position, velocity, orientation, angular velocity and rpm. Some divergence is expected, because PhysX integrates differently from the oracle's semi-implicit Euler. It loops `decimation` oracle steps per env step. |

```bash
# from the repo root
uv run python scripts/isaac_lab/diff_test_physics.py
uv run python scripts/isaac_lab/diff_test_pid.py
uv run python scripts/isaac_lab/diff_test_acro.py
<isaac-python> -u scripts/isaac_lab/diff_test_trajectory.py --headless --num_steps 100
```

`teacher_sweep.sh` (needs Isaac) is not a parity test: it flies the LOS teacher alone through the handoff
(`--teacher los --teacher-check-only`) for a list of `cruise_gain stop_speed` pairs given in `CFGS`, and prints only the
`TEACHER CHECK` / `teacher flight` / `trace` lines. Use it to pick teacher settings before a long run.

The physics test validates only the force and torque **math**. PhysX's own rigid-body integration is covered by the
trajectory test.

Older smoke and debug scripts from the Isaac migration are in `deprecated/scripts/`.
