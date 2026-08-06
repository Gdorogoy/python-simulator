# guidance

## Purpose
The backbone-brain of the whole project that makes the interceptor learn

## Methods/Exposes

### `train.py`

- `ActorCritic` - shared actor-critic network (one hidden layer, ReLU)
  - `__init__` - builds the shared layer plus the actor-mean head, a learned log-std parameter, and the critic (value) head
  - `forward` - runs `obs` through the shared layer, returns `(action_mean, action_std, state_value)`
  - `get_action_and_value` - builds a `Normal(action_mean, action_std)` distribution; samples an action if none is given (or scores a given one); returns `(action, log_prob, entropy, value)`. `entropy` measures how spread-out the action distribution is and is used later as an exploration bonus in the loss, not an error metric.
- `RolloutBuffer` - fixed-size storage for one rollout's worth of transitions
  - `__init__` - preallocates zero tensors for `obs`, `actions`, `log_probs`, `rewards`, `values`, `dones`, plus a write pointer `ptr`
  - `add` - writes a single timestep's transition `(obs, action, log_prob, reward, value, done)` into the buffer at `ptr` and advances it, for later use in `ppo_update`
- `compute_gae` - generalized advantage estimation: how much better each action turned out to be than the value function expected, computed backwards through the rollout
- `ppo_update` - runs the PPO clipped-objective update: normalizes advantages, then for `num_epochs` over shuffled minibatches computes policy loss (clipped ratio), value loss, and an entropy bonus, and steps the optimizer
- `ppo_train` - main training loop: repeatedly collects a rollout with the current policy, computes GAE via `compute_gae`, updates the model via `ppo_update`, until `total_timestamps` is reached; returns the trained model and per-episode reward history
- `evaluate` - runs `n_episodes` deterministically (uses the distribution mean, not a sampled action) and prints the fraction of episodes that reached a reward `>= 15.0`

## Depends on
`torch`

## Notes

