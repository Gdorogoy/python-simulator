# app

The project package. Each folder has its own `docs.md` with purpose, files, core design and gotchas.

| Folder | Role |
|---|---|
| `dynamics/` | Physics engine: airframe parameters (`drone_params.py`), true state, numpy integrators (angle and acro style) and their batched torch mirrors. |
| `environmental/` | RL environments: numpy Gymnasium env (replay/eval), Isaac Lab env (training, angle and acro gym ids), chasing-drone model. |
| `control/` | PID controller, closed-form gain tuning, demonstrations / BC / DAgger, and the two-phase PID->RL agent. |
| `reward_functions/` | Rewards: `angle_reward_qv2`, `angle_reward_rc` (threats, zone hold), `acro_reward_v1`, and the shared termination geometry. |
| `guidance/` | Policy network and PPO core, checkpoint replay and recording, web viewer, plots, ONNX export. |
| `training/` | Trainers (`angle_rtl_train_isaac.py` for the angle drone, `acro_rtl_train_isaac.py` for the acro drone), evaluation pairs and diagnostics. |
| `navigation/` | Kalman state estimator (standalone, not yet wired into observations). |
| `test/` | Physics/config validation suite plus pytest unit tests for the rewards, threats and checkpoint loading. |

## Data flow

```
dynamics  <--  environmental (numpy + Isaac)  <--  reward_functions
                    |                    \
                    v                     v
              control (PID teacher)    guidance (ActorCritic, PPO)
                    \                     /
                     training (angle_rtl_train_isaac / acro_rtl_train_isaac) --> runs/<dir>/best.pt
                                                        |
                              guidance/record_run + serve_run (replay in the numpy env)
```

Two rules hold everywhere:
- Only `dynamics` changes the true state.
- The numpy implementations (`dynamics/angle_style_methods.py`, `control/pid.py`) are the reference. The torch / Isaac versions
  are diff-tested against them (`scripts/`).
