# app

The project package. Each folder has its own `docs.md` with purpose, files, core design and gotchas.

| Folder | Role |
|---|---|
| `dynamics/` | Physics engine: airframe config, true state, numpy integrator and its batched torch mirror. |
| `environmental/` | RL environments: numpy Gymnasium env (replay/eval) and Isaac Lab env (training). |
| `control/` | PID controller, closed-form gain tuning, demonstrations / BC / DAgger, and the two-phase PID->RL agent. |
| `reward_functions/` | Rewards: `reward_qv2` (current) and the shared termination geometry. |
| `guidance/` | Policy network and PPO core, checkpoint replay and recording, web viewer, plots, ONNX export. |
| `training/` | Trainers (`rtl_train_isaac.py` is current), evaluation pairs and diagnostics. |
| `navigation/` | Kalman state estimator (standalone, not yet wired into observations). |
| `test/` | Physics/config validation suite. |
| `acro_style/` | Placeholder packages for a planned racing-style project (`acro_style/docs/`). |

## Data flow

```
dynamics  <--  environmental (numpy + Isaac)  <--  reward_functions
                    |                    \
                    v                     v
              control (PID teacher)    guidance (ActorCritic, PPO)
                    \                     /
                     training (rtl_train_isaac) --> runs/<dir>/best.pt
                                                        |
                              guidance/record_run + serve_run (replay in the numpy env)
```

Two rules hold everywhere:
- Only `dynamics` changes the true state.
- The numpy implementations (`dynamics/methods.py`, `control/pid.py`) are the reference. The torch / Isaac versions
  are diff-tested against them (`scripts/`).
