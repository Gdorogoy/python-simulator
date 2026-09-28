# Optuna Search Space Audit

Covers `optuna_search.py` (base) and `optuna_search_isaac.py` (Isaac Lab).
Both share the same `suggest_params` logic; the Isaac version additionally
pins `hidden=64`, `num_hidden_layers=4`, `num_epochs=10`, `num_minibatches=64`.

---

## Missing parameter

### `log_std_min` — not searched

`ActorCritic` accepts both `log_std_min` (hardcoded default `-3.0`) and
`log_std_max` (searched over `[-2.5, -0.05]`). The effective action std is:

```
log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (tanh(actor_log_std) + 1)
```

The *window width* is `log_std_max - log_std_min`. When the sampler draws
`log_std_max` near its lower bound (`-2.5`), the window collapses to
`[-3.0, -2.5]` — a 0.5-unit range in log space. That confines the drone's
action std to roughly `[exp(-3.0), exp(-2.5)] = [0.05, 0.08]`, nearly
deterministic. The search never sees this because `log_std_min` is invisible
to Optuna.

**Fix (pick one):**
- Search `log_std_min` jointly, e.g. `suggest_float("log_std_min", -4.0, -1.5)`,
  and enforce `log_std_min < log_std_max - 0.5` after the draw.
- Or raise `log_std_max`'s lower bound from `-2.5` to `-1.5` so the window
  can never collapse below 1.5 log-units (~std range 0.22 to whatever max is).

---

## Ranges with wasted trial budget

### `lr` — upper bound too high

| | Value |
|---|---|
| Current range | `[1e-6, 1e-2]` (log scale) |
| PARAMS default | `5e-5` |
| Typical PPO sweet spot | `1e-5` – `3e-4` |

The upper bound `1e-2` is 200× the default. PPO with AdamW rarely survives
above ~`3e-4` without gradient explosion; anything above ~`1e-3` will either
fail outright or get pruned early. These trials still consume their warmup
budget before the pruner can act.

**Suggested range:** `[1e-5, 5e-4]`

---

### `max_grad_norm` — upper bound too permissive

| | Value |
|---|---|
| Current range | `[0.1, 2.0]` |
| PARAMS default | `0.25` |

Values above ~`1.0` are unusually permissive for PPO and allow loss spikes
that the clip ratio alone won't contain. The default sits near the tight end
of the current range, suggesting the upper half rarely pays off.

**Suggested range:** `[0.1, 1.0]`

---

### `num_epochs` — upper bound too high *(base search only)*

| | Value |
|---|---|
| Current range | `[3, 20]` |
| PARAMS default | `10` |

More than ~12 PPO update epochs per rollout frequently causes policy collapse
because the same minibatch data is reused until the ratio clips heavily.
Pinned at `10` in the Isaac search; the base search should cap lower.

**Suggested range:** `[3, 15]`

---

### `vf_coef` — upper bound slightly high

| | Value |
|---|---|
| Current range | `[0.1, 1.5]` |
| PARAMS default | `0.525` |

At `1.5` the value loss dominates the policy gradient. Minor waste compared
to `lr`, but `[0.1, 1.0]` covers all practically useful values.

**Suggested range:** `[0.1, 1.0]`

---

## Ranges that look fine

| Param | Range | Notes |
|---|---|---|
| `lam` | `[0.85, 0.999]` | Covers full GAE spectrum; default 0.97 sits well inside. |
| `clip_eps` | `[0.05, 0.4]` | Standard PPO range. |
| `target_kl` | `[0.005, 0.15]` log | Sensible; log scale is correct here. |
| `ent_coef_start` | `[1e-4, 0.1]` log | Good. |
| `ent_coef_end_frac` | `[0.01, 0.9]` | Fraction-of-start design avoids inverted decay. |
| `weight_decay` | `[1e-6, 1e-2]` log | Standard AdamW range. |
| `lr_min_ratio` | `[0.0, 0.5]` | Linear is correct (0.0 is a valid "fully decay" setting). |
| `log_std_max` | `[-2.5, -0.05]` | Fine in isolation; problem is the interaction with fixed `log_std_min` (see above). |

---

## Action items (priority order)

1. **Search or constrain `log_std_min`** — correctness issue, not just waste.
2. **Tighten `lr` upper bound** to `5e-4` — biggest source of wasted trials.
3. **Tighten `max_grad_norm`** to `1.0` and **`num_epochs`** to `15` — easy wins.
4. **Tighten `vf_coef`** to `1.0` — minor, address with the others.
