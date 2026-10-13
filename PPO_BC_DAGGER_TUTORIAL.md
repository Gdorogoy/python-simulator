# PPO + BC + DAgger — one-read tutorial

Corrections to common restatements:

- **BC** is not stored data that gets loaded. It is a supervised training run: the NN's output is regressed onto the PID's action for each recorded state.
- **DAgger** does not run the NN "during imitation". It is a data-collection loop: the NN drives, the PID labels the states the NN reached, and the NN retrains on the growing dataset.
- Neither is "loaded with the NN". They produce the NN's starting weights. The `distill` hyperparameter is the weight of an imitation loss kept inside the PPO loss and decayed over time.

## 1. PPO

### Setup
- **Actor:** outputs a mean action and has a learned `log_std`. During training, actions are sampled from N(mean, std). `π_θ(a|s)` is the probability density of the sampled action.
- **Critic:** `V_φ(s)` predicts the discounted future reward from state `s`.

### Loop
1. **Collect.** Run the current policy `π_old` for N steps across the envs. Store `s, a, r, done, logπ_old(a|s), V(s)`.
2. **Compute advantages with GAE.** It blends one-step and full-episode estimates:
   ```
   δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
   A_t = δ_t + γ·λ·(1−done_t)·A_{t+1}      (computed backwards)
   R_t = A_t + V(s_t)                        (critic's target)
   ```
   - `γ` is about 0.99.
   - `λ` is about 0.95. `λ=0` is pure one-step TD (low variance, biased). `λ=1` is pure Monte Carlo (unbiased, noisy).
   - `A_t > 0` means this action turned out better than the critic expected. `A_t < 0` means worse.
   - Normalize `A` to mean 0 and std 1 per batch.
3. **Update for K epochs over shuffled minibatches of the same data.**
   ```
   ratio = exp(logπ_θ(a|s) − logπ_old(a|s))
   L_actor = −E[ min( ratio·A, clip(ratio, 1−ε, 1+ε)·A ) ]
   L_critic = E[ (V_φ(s) − R_t)² ]
   L_total = L_actor + 0.5·L_critic − c_ent·entropy
   ```
   - **Clip logic:** if `A>0`, the push to raise the action's probability stops once `ratio > 1+ε`. If `A<0`, the push to lower it stops once `ratio < 1−ε`. The `min` takes the more pessimistic of the two terms. So a single batch can't drag the policy far, even though it is reused K times. This is PPO's cheap version of a trust region.
   - **Typical values:** `ε=0.2`, `K=4–10`, lr about 3e-4, grad-clip 0.5, `c_ent` 0–0.01.
4. Set `π_old = π_θ` and go back to step 1.

### Details that matter
- **Termination vs time-limit.** On a crash (a terminal state), the future value is 0. On an episode time-out, bootstrap with `V(s_next)`. Mixing these up corrupts the advantages.
- **Reward and observation scaling.** Normalize both, or the critic's loss scale will dominate the actor's.
- **Reward hacking** is the policy maximizing your formula instead of your intent. Farming reward at ±0.2 m from the target is the classic case. The fix is to make staying near the target without hitting it worse than trying again (an escalating per-step penalty, or a bonus only for a real hit).

### Diagnostics to log

| Metric | Healthy | If wrong |
|---|---|---|
| `approx_kl` | about 0.01–0.02 | Too high means the policy moves too fast, so lower lr or `K`. |
| `clip_fraction` | 0.1–0.3 | Near 0 means updates are too small. Near 1 means they're too big. |
| `explained_variance` of V | rising toward 1 | Near 0 means advantages are noise. |
| `std` / entropy | slow decline | Collapsing early means no exploration. |
| Episode return and length | rising | |

## 2. BC (behavior cloning)

- **Data:** pairs `(s, a_pid)` recorded while the PID flies.
- **Training:** minimize `E‖π_θ(s) − a_pid‖²`. This is plain supervised learning, with no rewards.
- **Weakness (covariate shift):** the NN is trained only on states the PID visits. Its small errors push it into states the PID never visited, where it has no data, so errors compound. The error grows roughly with the horizon squared.

## 3. DAgger

```
D = BC dataset
repeat:
    run policy π_i = β_i·PID + (1−β_i)·NN      (β_i shrinks toward 0)
    at every visited state s: label = a_pid(s)  (the teacher always labels, even when the NN acted)
    D = D ∪ {(s, label)}
    retrain NN on D
```
- Training now covers the states the NN actually drifts into, so the error grows only linearly with the horizon.
- It needs a teacher you can query for any state. The PID qualifies.

## 4. How they combine (teacher-student)

1. **BC, then DAgger,** produce the initial actor weights.
2. **Warm up the critic before PPO touches the actor.** Keep the actor frozen and train only `V_φ` on rollouts until `explained_variance` is positive. A random critic gives garbage advantages, and PPO then walks a good actor away from where it started. This is a hypothesis for the drift seen after unfreezing (v1–v3), not confirmed against the code.
3. **Start `log_std` small but not collapsed.** A large std swamps the BC mean.
4. **PPO with an annealed imitation term:**
   ```
   L = L_PPO + distill(t)·E_s~visited ‖π_θ(s) − a_pid(s)‖²
   ```
   - `distill` decays toward 0, so the loss anchors the actor to the PID early and lets reward take over later. This is the "RTL" part.
   - It is DAgger-like if `s` comes from the NN's own rollouts and the PID labels them.
5. **Pitfall with an α-blend.** If the executed action is `α·nn + (1−α)·pid`, the stored `logπ_old` must be for the NN's own sampled action. Otherwise the ratio `exp(logπ_θ − logπ_old)` is computed for an action the NN didn't take. That biases or breaks the PPO update. Check the code for this.

## 5. Why the PID→NN handoff works

The NN works at takeover because the observations it sees (relative target position, speed range) match what it was trained on. Resetting the coordinates so the NN navigates from x=y=0 is what keeps the observations in-distribution. Without that, it would fly the right path in the wrong x or y.
